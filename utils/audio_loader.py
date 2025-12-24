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
        DeepFilterNet3 natively supports 48kHz processing.

        Args:
            audio_data: Audio data as bytes (WAV format)
            vo_threshold: Voice detection threshold value
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

            # DeepFilterNet processing with CPU only - using native 48kHz
            target_sample_rate = 48000
            
            logger.info(f"🎙️ [VO] 🚀 Starting DeepFilterNet processing on CPU")
            logger.info(f"🎙️ [VO] Using sample rate: {target_sample_rate}Hz")
            
            try:
                # First, capture the original sample rate from input file
                info = torchaudio.info(input_file)
                original_input_sr = info.sample_rate
                logger.info(f"🎙️ [VO] Original input sample rate: {original_input_sr}Hz")
                
                # Load and process audio on CPU
                wav, meta = load_audio(input_file, target_sample_rate)
                
                # Ensure mono processing
                if wav.shape[0] > 1:
                    wav = torch.mean(wav, dim=0, keepdim=True)
                    logger.info(f"🎙️ [VO] Converted to mono for processing")
                
                # Log tensor details
                tensor_memory_mb = wav.numel() * wav.element_size() / (1024**2)
                logger.info(f"🎙️ [VO] Audio tensor loaded: {tensor_memory_mb:.1f} MB memory (sample_rate={target_sample_rate}Hz, channels={wav.shape[0]})")
                
                enhanced = enhance(model, df_state, wav, pad=True)
                
                # Save with the target sample rate
                torchaudio.save(output_file, enhanced, target_sample_rate)
                
                # Cleanup
                del wav, enhanced
                    
                logger.info(f"🎙️ [VO] ✅ CPU processing completed successfully")
                
            except Exception as processing_error:
                logger.error(f"🎙️ [VO] ❌ CPU processing failed: {processing_error}")
                logger.warning(f"🎙️ [VO] 🟡 Processing failed, returning original audio")
                return audio_data, False, -1.0

            # Check if output file exists
            if not os.path.exists(output_file):
                logger.error(f"Error: Output file {output_file} not found")
                raise Exception(f"Output file {output_file} not found")

            # Load both files for noise analysis
            original_waveform, original_sr = torchaudio.load(input_file)
            enhanced_waveform, enhanced_sr = torchaudio.load(output_file)

            # Ensure same sample rate - resample original to target_sample_rate to match processing domain
            if original_sr != target_sample_rate:
                logger.info(f"Resampling original audio from {original_sr}Hz to {target_sample_rate}Hz for noise analysis")
                resampler = torchaudio.transforms.Resample(original_sr, target_sample_rate)
                original_waveform = resampler(original_waveform)

            # Enhanced should already be at target_sample_rate, verify
            if enhanced_sr != target_sample_rate:
                logger.warning(f"Enhanced audio is at {enhanced_sr}Hz, expected {target_sample_rate}Hz. Resampling to target rate.")
                resampler = torchaudio.transforms.Resample(enhanced_sr, target_sample_rate)
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

            # Calculate noise estimate (at 48kHz)
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
                logger.info(f"VO Signal Ratio ({noise_ratio2:.4f}) exceeds threshold ({vo_threshold}), returning original audio")
                return audio_data, False, noise_ratio2
            else:
                logger.info("🟢 [VO] returning processed audio")
                logger.info(f"VO Signal Ratio ({noise_ratio2:.4f}) is below threshold ({vo_threshold})")
                
                # Resample noise_estimate back to original input sample rate if needed
                if target_sample_rate != original_input_sr:
                    logger.info(f"Resampling noise estimate from {target_sample_rate}Hz back to original {original_input_sr}Hz")
                    resampler = torchaudio.transforms.Resample(target_sample_rate, original_input_sr)
                    noise_estimate = resampler(noise_estimate)
                    output_sr = original_input_sr
                else:
                    output_sr = target_sample_rate
                    logger.info(f"Audio already at target output rate: {output_sr}Hz")
                
                # Save the processed audio to a temp file
                processed_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                torchaudio.save(processed_file, noise_estimate, output_sr)
                
                # Read processed audio file and return as bytes
                with open(processed_file, "rb") as f:
                    processed_audio_data = f.read()
                
                # Clean up temp file
                if os.path.exists(processed_file):
                    os.unlink(processed_file)

                logger.info(f"🎙️ [VO] Returning audio at {output_sr}Hz (from original {original_input_sr}Hz)")
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
            logger.debug(f"🎙️ [VO] CPU processing cleanup completed")
