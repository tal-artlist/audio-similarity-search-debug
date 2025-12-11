"""
Simplified AudioLoader for standalone audio similarity search.
This version only includes basic functionality needed for the similarity search.
"""
import gc
import os
import shutil
import tempfile
import logging
from io import BytesIO
from typing import Tuple
import numpy as np
import torch
import torchaudio
from pydub import AudioSegment

logger = logging.getLogger(__name__)

class AudioLoader:
    def __init__(self, download_folder: str = "/tmp/audio_downloads"):
        self.download_folder = download_folder

    def get_audio_duration(self, file_path: str) -> float:
        """Get duration of audio file in seconds"""
        audio = AudioSegment.from_file(file_path)
        duration = len(audio) / 1000.0  # Convert milliseconds to seconds
        del audio
        gc.collect()
        return duration
    
    def process_audio_with_vo_threshold(
        self, audio_data: bytes, vo_threshold: float = 0.5, df_model_tuple=None
    ) -> Tuple[bytes, bool, float]:
        """
        Process audio data with VO threshold using DeepFilterNet3.

        Args:
            audio_data: Audio data as bytes (WAV format)
            vo_threshold: Voice activity threshold value
            df_model_tuple: Pre-loaded DeepFilterNet model tuple (model, df_state, device)

        Returns:
            Tuple[bytes, bool, float]: (processed_audio_data, is_voice_detected, signal_ratio)
                - processed_audio_data: Processed or original audio as bytes
                - is_voice_detected: True if voice is detected (noise ratio below threshold), False otherwise
                - signal_ratio: The calculated VO signal ratio value
        """
        from df.enhance import enhance, load_audio
        
        logger.info(f"🎙️ [VO] process_audio_with_vo_threshold: start threshold={vo_threshold}")
        input_file = None
        output_file = None

        try:
            # Load audio from bytes
            audio = AudioSegment.from_file(BytesIO(audio_data))

            # Save audio data to temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(audio_data)
                input_file = temp_file.name

            # Create output directory in the same temp location
            input_filename = os.path.basename(input_file)
            input_filename_base = os.path.splitext(input_filename)[0]
            temp_output_dir = os.path.join(os.path.dirname(input_file), "audio_files")
            os.makedirs(temp_output_dir, exist_ok=True)
            
            # deepFilter outputs to: {output_dir}/{input_filename_base}_DeepFilterNet3_pf.wav
            output_file = os.path.join(temp_output_dir, f"{input_filename_base}_DeepFilterNet3_pf.wav")
            
            # Use pre-loaded DeepFilterNet model from startup
            if df_model_tuple is None:
                raise ValueError("DeepFilterNet model not provided. Ensure the model is loaded at startup.")
            
            model, df_state, device = df_model_tuple

            # DeepFilterNet processing with device auto-detection
            target_sample_rate = 44100
            
            # Detect available processing device
            cuda_available = torch.cuda.is_available() and device == "cuda"
            processing_device = "GPU" if cuda_available else "CPU"
            
            logger.info(f"🎙️ [VO] 🚀 Starting DeepFilterNet processing on {processing_device}")
            logger.info(f"🎙️ [VO] Using sample rate: {target_sample_rate}Hz")
            
            try:
                # Load and process audio
                wav, meta = load_audio(input_file, target_sample_rate, device)
                
                # Ensure mono processing
                if wav.shape[0] > 1:
                    wav = torch.mean(wav, dim=0, keepdim=True)
                    logger.info(f"🎙️ [VO] Converted to mono for processing")
                
                # Log tensor details
                tensor_memory_mb = wav.numel() * wav.element_size() / (1024**2)
                if cuda_available:
                    logger.info(f"🎙️ [VO] Audio tensor loaded: {tensor_memory_mb:.1f} MB GPU memory (sample_rate={target_sample_rate}Hz, channels={wav.shape[0]})")
                else:
                    logger.info(f"🎙️ [VO] Audio tensor loaded: {tensor_memory_mb:.1f} MB memory (sample_rate={target_sample_rate}Hz, channels={wav.shape[0]})")
                
                enhanced = enhance(model, df_state, wav, pad=True)
                
                # Save with the target sample rate
                torchaudio.save(output_file, enhanced.to("cpu"), target_sample_rate)
                
                # Cleanup
                del wav, enhanced
                if cuda_available:
                    torch.cuda.empty_cache()
                    
                logger.info(f"🎙️ [VO] ✅ {processing_device} processing completed successfully")
                
            except Exception as processing_error:
                logger.warning(f"🎙️ [VO] ❌ {processing_device} processing failed: {processing_error}")
                
                # Clean up any partial tensors
                try:
                    if 'wav' in locals():
                        del wav
                    if 'enhanced' in locals():
                        del enhanced
                    if cuda_available:
                        torch.cuda.empty_cache()
                except:
                    pass
                
                # If GPU failed, try CPU fallback. If CPU failed, return original.
                if cuda_available:
                    logger.warning(f"🎙️ [VO] ⚠️ GPU processing failed, trying CPU fallback")
                    
                    try:
                        logger.info(f"🎙️ [VO] 🖥️ CPU fallback: Starting DeepFilterNet processing")
                        
                        # Clean up GPU memory before CPU fallback
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        logger.info(f"🎙️ [VO] 🖥️ GPU memory cleared")
                        
                        # Initialize fresh CPU-only model
                        logger.info(f"🎙️ [VO] 🖥️ Initializing CPU-only model")
                        from df import init_df
                        
                        # Block CUDA for CPU-only processing
                        original_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
                        os.environ['CUDA_VISIBLE_DEVICES'] = ''
                        logger.info(f"🎙️ [VO] 🖥️ CUDA blocked for CPU processing")
                        
                        try:
                            # Initialize CPU model
                            cpu_model, cpu_df_state, _ = init_df(model_base_dir="DeepFilterNet3")
                            cpu_model = cpu_model.eval()
                            logger.info(f"🎙️ [VO] 🖥️ CPU model initialized successfully")
                            
                            # Process on CPU
                            wav_cpu, meta_cpu = load_audio(input_file, target_sample_rate)
                            
                            # Ensure mono
                            if wav_cpu.shape[0] > 1:
                                wav_cpu = torch.mean(wav_cpu, dim=0, keepdim=True)
                                logger.info(f"🎙️ [VO] 🖥️ Converted to mono")
                            
                            logger.info(f"🎙️ [VO] 🖥️ Audio loaded on CPU")
                            
                            # Log CPU processing details
                            tensor_memory_mb = wav_cpu.numel() * wav_cpu.element_size() / (1024**2)
                            logger.info(f"🎙️ [VO] 🖥️ CPU tensor: {tensor_memory_mb:.1f} MB (sample_rate={target_sample_rate}Hz, channels={wav_cpu.shape[0]})")
                            
                            # Enhance on CPU
                            logger.info(f"🎙️ [VO] 🖥️ Running DeepFilterNet enhancement")
                            with torch.no_grad():
                                enhanced_cpu = enhance(cpu_model, cpu_df_state, wav_cpu, pad=True)
                            
                            # Save result
                            torchaudio.save(output_file, enhanced_cpu, target_sample_rate)
                            
                            # Cleanup
                            del wav_cpu, enhanced_cpu
                            
                            logger.info(f"🎙️ [VO] ✅ CPU processing completed successfully")
                            
                        finally:
                            # Restore CUDA visibility
                            if original_cuda_visible_devices is not None:
                                os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible_devices
                            else:
                                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                            logger.info(f"🎙️ [VO] 🖥️ CUDA visibility restored")
                            
                    except Exception as cpu_error:
                        logger.error(f"🎙️ [VO] ❌ CPU fallback also failed: {cpu_error}")
                        logger.warning(f"🎙️ [VO] 🟡 Both GPU and CPU processing failed, returning original audio")
                        return audio_data, False, -1.0
                else:
                    # CPU processing failed and no GPU available
                    logger.error(f"🎙️ [VO] ❌ CPU processing failed, no fallback available")
                    logger.warning(f"🎙️ [VO] 🟡 Processing failed, returning original audio")
                    return audio_data, False, -1.0

            # Check if output file exists
            if not os.path.exists(output_file):
                logger.error(f"Error: Output file {output_file} not found")
                raise Exception(f"Output file {output_file} not found")

            # Load both files for noise analysis
            original_waveform, original_sr = torchaudio.load(input_file)
            enhanced_waveform, enhanced_sr = torchaudio.load(output_file)

            # Ensure same sample rate
            if original_sr != enhanced_sr:
                logger.info(
                    f"Resampling enhanced audio from {enhanced_sr}Hz to {original_sr}Hz"
                )
                resampler = torchaudio.transforms.Resample(enhanced_sr, original_sr)
                enhanced_waveform = resampler(enhanced_waveform)

            # Ensure both are mono
            if original_waveform.shape[0] > 1:
                original_waveform = torch.mean(original_waveform, dim=0, keepdim=True)
                logger.info("Converted original to mono")

            if enhanced_waveform.shape[0] > 1:
                enhanced_waveform = torch.mean(enhanced_waveform, dim=0, keepdim=True)
                logger.info("Converted enhanced to mono")

            # Ensure same length
            min_length = min(original_waveform.shape[1], enhanced_waveform.shape[1])
            original_waveform = original_waveform[:, :min_length]
            enhanced_waveform = enhanced_waveform[:, :min_length]

            # Calculate noise estimate
            noise_estimate = original_waveform - enhanced_waveform
            
            weights = np.abs(enhanced_waveform.numpy())
            weights /= np.max(weights) if np.max(weights) > 0 else 1.0
            noise_power2 = np.average(noise_estimate.numpy() ** 2, weights=weights)
            signal_power2 = np.average(enhanced_waveform.numpy() ** 2, weights=weights)
            noise_ratio2 = (
                noise_power2 / (noise_power2 + signal_power2)
                if (noise_power2 + signal_power2) > 0
                else 0.0
            )

            logger.info(f"VO Signal Ratio: {noise_ratio2:.4f}")

            # Check if noise ratio exceeds threshold
            if noise_ratio2 > vo_threshold:
                logger.info("🟡 [VO] returning original audio (threshold exceeded)")
                logger.info(
                    f"VO Signal Ratio ({noise_ratio2:.4f}) exceeds threshold ({vo_threshold}), returning original audio"
                )
                return audio_data, False, noise_ratio2
            else:
                logger.info("🟢 [VO] returning processed audio")
                logger.info(
                    f"VO Signal Ratio ({noise_ratio2:.4f}) is below threshold ({vo_threshold})"
                )
                # Save the processed audio (noise estimate) to a temp file
                processed_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                torchaudio.save(processed_file, noise_estimate, original_sr)
                
                # Read processed audio file and return as bytes
                with open(processed_file, "rb") as f:
                    processed_audio_data = f.read()
                
                # Clean up temp file
                if os.path.exists(processed_file):
                    os.unlink(processed_file)

                return processed_audio_data, True, noise_ratio2

        except Exception as e:
            logger.error(f"Failed to process audio with VO threshold: {e}")
            raise Exception(f"Audio processing failed: {e}")
        finally:
            # Clean up temporary files
            if input_file and os.path.exists(input_file):
                os.unlink(input_file)
            if output_file and os.path.exists(output_file):
                os.unlink(output_file)
            # Clean up temporary output directory
            if 'temp_output_dir' in locals() and os.path.exists(temp_output_dir):
                shutil.rmtree(temp_output_dir, ignore_errors=True)
            
            # Cleanup device memory if needed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug(f"🎙️ [VO] GPU memory cleanup completed")
