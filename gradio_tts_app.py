import random
import numpy as np
import torch
import gradio as gr
from chatterbox.tts import ChatterboxTTS
import time
import os
import logging
import re
import asyncio
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_model():
    """Load the ChatterboxTTS model."""
    model = ChatterboxTTS.from_pretrained(DEVICE)
    logger.info(f"Loaded ChatterboxTTS model on device: {DEVICE}")
    return model

def chunk_text(text, max_chunk_size=300):
    """Split text into smaller chunks at sentence boundaries."""
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text.strip())
    
    if not text:
        return []

    # Split on sentence delimiters while preserving the delimiter
    delimiter_pattern = r'(?<=[.!?])\s+'
    segments = re.split(delimiter_pattern, text)

    # Process segments to ensure each has appropriate ending punctuation
    sentences = []
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # Check if segment already ends with a delimiter
        if not segment[-1] in ['.', '!', '?']:
            segment += '.'

        sentences.append(segment)
    
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence would make the chunk too long, start a new chunk
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence

    # Add the last chunk if there's anything left
    if current_chunk:
        chunks.append(current_chunk)

    logger.info(f"Text chunked into {len(chunks)} segments")
    return chunks

def generate_single_speech(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfg_weight):
    """Generate speech for single text input using ChatterboxTTS."""
    if model is None:
        model = ChatterboxTTS.from_pretrained(DEVICE)
    
    if not text.strip():
        return None, "Error: No text provided"
    
    try:
        start_time = time.monotonic()
        
        if seed_num != 0:
            set_seed(int(seed_num))
        
        # Generate speech using actual ChatterboxTTS API
        wav = model.generate(
            text,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
        )
        
        # Calculate stats
        duration = wav.shape[-1] / model.sr
        processing_time = time.monotonic() - start_time
        result_message = f"Generated {duration:.2f} seconds of audio in {processing_time:.2f} seconds"
        
        # Return in Gradio audio format (sample_rate, numpy_array)
        return (model.sr, wav.squeeze(0).numpy()), result_message
        
    except Exception as e:
        error_msg = f"Error generating speech: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

def generate_chunk_audio(model, chunk, audio_prompt_path, exaggeration, temperature, cfg_weight, temp_dir, chunk_idx):
    """Generate audio for a single chunk."""
    try:
        # Generate audio for this chunk
        wav = model.generate(
            chunk,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
        )
        
        # Save as numpy file for later concatenation
        chunk_filename = os.path.join(temp_dir, f"chunk_{chunk_idx:03d}.npy")
        np.save(chunk_filename, wav.squeeze(0).numpy())
        
        # Calculate duration
        duration = wav.shape[-1] / model.sr
        
        return chunk_filename, duration, model.sr
        
    except Exception as e:
        logger.error(f"Error generating chunk {chunk_idx}: {str(e)}")
        raise

async def process_chunk_async(model, chunk, audio_prompt_path, exaggeration, temperature, cfg_weight, temp_dir, chunk_idx, total_chunks):
    """Process a single chunk asynchronously."""
    loop = asyncio.get_event_loop()
    
    # Run the model inference in a separate thread since it's blocking
    def generate_for_chunk():
        return generate_chunk_audio(model, chunk, audio_prompt_path, exaggeration, temperature, cfg_weight, temp_dir, chunk_idx)
    
    # Execute the model inference in a thread
    filename, duration, sample_rate = await loop.run_in_executor(None, generate_for_chunk)
    
    logger.info(f"Processed chunk {chunk_idx + 1}/{total_chunks}: {duration:.2f}s")
    
    return filename, duration, sample_rate

async def generate_long_form_speech_async(model, long_text, audio_prompt_path, exaggeration, temperature, cfg_weight, seed_num, batch_size=4, max_chunk_size=300, progress=None):
    """Generate speech for long-form text with parallel processing."""
    start_time = time.monotonic()
    
    # Set seed if specified
    if seed_num != 0:
        set_seed(int(seed_num))
    
    if progress is not None:
        progress(0, desc="Preparing text chunks...")
    
    # Chunk the text
    chunks = chunk_text(long_text, max_chunk_size)
    
    if not chunks:
        raise ValueError("No valid text chunks found")
    
    if progress is not None:
        progress(0.1, desc=f"Text split into {len(chunks)} chunks")
    
    # Create a temporary directory for chunk files
    temp_dir = tempfile.mkdtemp(prefix="longform_tts_")
    logger.info(f"Created temp directory: {temp_dir}")
    
    try:
        # Use a semaphore to limit concurrent processing to batch_size
        semaphore = asyncio.Semaphore(batch_size)
        total_chunks = len(chunks)
        all_audio_files = []
        total_duration = 0
        processed_chunks = 0
        sample_rate = None

        async def process_chunk_with_semaphore(chunk, idx):
            nonlocal processed_chunks, total_duration, sample_rate
            async with semaphore:
                try:
                    filename, duration, sr = await process_chunk_async(
                        model, chunk, audio_prompt_path, exaggeration, temperature, cfg_weight, temp_dir, idx, total_chunks
                    )
                    processed_chunks += 1
                    total_duration += duration
                    
                    if sample_rate is None:
                        sample_rate = sr
                    
                    if progress is not None:
                        progress_val = 0.1 + (processed_chunks / total_chunks) * 0.8
                        progress(progress_val, 
                               desc=f"Processed chunk {processed_chunks}/{total_chunks}")
                    
                    return filename, duration
                except Exception as e:
                    logger.error(f"Error processing chunk {idx}: {str(e)}")
                    raise

        # Create tasks for all chunks and process them concurrently
        tasks = [process_chunk_with_semaphore(chunk, idx) 
                for idx, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)

        # Sort results by chunk index and collect filenames
        sorted_results = sorted(results, key=lambda x: int(os.path.basename(x[0]).split('_')[1].split('.')[0]))
        all_audio_files = [result[0] for result in sorted_results]

        # Combine all audio files
        if progress is not None:
            progress(0.9, desc="Combining audio files...")

        logger.info(f"Combining {len(all_audio_files)} audio chunks")

        # Load and concatenate all audio files
        combined_audio = []
        
        for audio_file in all_audio_files:
            chunk_audio = np.load(audio_file)
            combined_audio.append(chunk_audio)

        # Concatenate all audio
        if combined_audio:
            final_audio = np.concatenate(combined_audio)
        else:
            raise ValueError("No audio data to combine")

        # Calculate final stats
        processing_time = time.monotonic() - start_time
        result_message = (f"Generated {total_duration:.2f} seconds of audio from "
                         f"{total_chunks} chunks in {processing_time:.2f} seconds")
        logger.info(result_message)
        
        if progress is not None:
            progress(1.0, desc="Complete!")

        # Return in Gradio audio format (sample_rate, numpy_array)
        return (sample_rate, final_audio), result_message

    finally:
        # Clean up temporary files
        try:
            for file in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")

def generate_long_form_speech(model, long_text, audio_prompt_path, exaggeration, temperature, cfg_weight, seed_num, batch_size=4, max_chunk_size=300, progress=gr.Progress()):
    """Wrapper function for Gradio interface - Long Form."""
    if model is None:
        model = ChatterboxTTS.from_pretrained(DEVICE)
        
    if not long_text.strip():
        return None, "Error: No text provided"
    
    try:
        # Run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                generate_long_form_speech_async(
                    model, long_text, audio_prompt_path, exaggeration, temperature, cfg_weight, seed_num, batch_size, max_chunk_size, progress
                )
            )
        finally:
            loop.close()
            
    except Exception as e:
        error_msg = f"Error generating long-form speech: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

def cleanup_files():
    """Clean up generated audio files - for demo.load()."""
    count = 0
    for file in os.listdir():
        if (file.startswith("output_") or file.startswith("longform_output_")) and file.endswith(".wav"):
            try:
                os.remove(file)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete file {file}: {e}")
    
    logger.info(f"Cleanup completed. Removed {count} files.")

def cleanup_files_with_result():
    """Clean up generated audio files - for button click with result."""
    count = 0
    for file in os.listdir():
        if (file.startswith("output_") or file.startswith("longform_output_")) and file.endswith(".wav"):
            try:
                os.remove(file)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete file {file}: {e}")
    
    logger.info(f"Cleanup completed. Removed {count} files.")
    return f"Cleaned up {count} files."

def create_gradio_interface():
    """Create the Gradio interface with both single and long-form tabs."""
    with gr.Blocks(title="Chatterbox-TTS-WebUI", theme=gr.themes.Default()) as demo:
        
        gr.Markdown("<div align='center'><h1>Chatterbox-TTS-WebUI</h1></div>")
        gr.Markdown("<div align='center'>Generate realistic speech from text with voice cloning capabilities</div>")
        
        # Model state - loaded once per session
        model_state = gr.State(None)
        
        with gr.Tabs() as tabs:
            # Tab 1: Single Text Generation
            with gr.Tab("Single Text"):
                with gr.Row():
                    with gr.Column(scale=2):
                        single_text = gr.Textbox(
                            value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                            label="Text to synthesize (max chars 300)",
                            lines=5,
                            max_lines=5
                        )
                        
                        single_ref_wav = gr.Audio(
                            sources=["upload", "microphone"], 
                            type="filepath", 
                            label="Reference Audio File", 
                            value=None
                        )
                        
                        with gr.Row():
                            single_exaggeration = gr.Slider(
                                0.25, 2, step=.05, 
                                label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", 
                                value=.5
                            )
                            single_cfg_weight = gr.Slider(
                                0.0, 1, step=.05, 
                                label="CFG/Pace", 
                                value=0.5
                            )
                        
                        with gr.Accordion("More options", open=False):
                            single_seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                            single_temp = gr.Slider(0.05, 5, step=.05, label="Temperature", value=.8)
                        
                        single_submit = gr.Button("Generate", variant="primary")
                        
                        gr.Examples(
                            examples=[
                                "Hello, this is a test of the ChatterboxTTS system.",
                                "The quick brown fox jumps over the lazy dog.",
                                "Welcome to the voice cloning demonstration.",
                                "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
                            ],
                            inputs=single_text,
                            label="Example Texts"
                        )
                    
                    with gr.Column(scale=1):
                        single_audio = gr.Audio(label="Generated Speech")
                        single_result = gr.Textbox(label="Generation Stats", interactive=False)
                
                single_submit.click(
                    fn=generate_single_speech,
                    inputs=[model_state, single_text, single_ref_wav, single_exaggeration, single_temp, single_seed_num, single_cfg_weight],
                    outputs=[single_audio, single_result]
                )
            
            # Tab 2: Long Form Content
            with gr.Tab("Long Form Content"):
                with gr.Row():
                    with gr.Column(scale=2):
                        long_text = gr.Textbox(
                            label="Long Form Text Input",
                            placeholder="Enter long text to convert to speech...",
                            lines=12
                        )
                        
                        long_ref_wav = gr.Audio(
                            sources=["upload", "microphone"], 
                            type="filepath", 
                            label="Reference Audio File (will be used for all chunks)", 
                            value=None
                        )
                        
                        with gr.Row():
                            long_exaggeration = gr.Slider(
                                0.25, 2, step=.05, 
                                label="Exaggeration", 
                                value=.5
                            )
                            long_cfg_weight = gr.Slider(
                                0.0, 1, step=.05, 
                                label="CFG/Pace", 
                                value=0.5
                            )
                            long_temp = gr.Slider(
                                0.05, 5, step=.05, 
                                label="Temperature", 
                                value=.8
                            )
                        
                        with gr.Accordion("Long Form Options", open=True):
                            with gr.Row():
                                batch_size = gr.Slider(
                                    minimum=1,
                                    maximum=8,
                                    value=4,
                                    step=1,
                                    label="Batch Size (parallel chunks)"
                                )
                                
                                max_chunk_size = gr.Slider(
                                    minimum=100,
                                    maximum=500,
                                    value=300,
                                    step=50,
                                    label="Max Chunk Size (characters)"
                                )
                        
                        with gr.Accordion("More options", open=False):
                            long_seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                        
                        long_submit = gr.Button("Generate Long Form Speech", variant="primary")
                        
                        gr.Markdown("""
                        ### How Long Form Processing Works:
                        * **Smart Text Chunking**: Automatically splits long text into smaller chunks at sentence boundaries
                        * **Parallel Processing**: Processes multiple chunks simultaneously for faster generation
                        * **Seamless Audio Stitching**: Combines multiple audio segments into one cohesive output file
                        * **Progress Tracking**: Real-time progress indicators during the generation process
                        * **Voice Consistency**: Uses the same reference audio for all chunks
                        """)
                        
                        gr.Examples(
                            examples=[
                                """This is a demonstration of long form text-to-speech generation with voice cloning. 
The system automatically splits this text into smaller chunks at sentence boundaries. 
Each chunk is processed in parallel for faster generation using the same cloned voice. 
Finally, all chunks are seamlessly combined into a single audio file. 
This approach allows for processing much longer texts efficiently while maintaining voice consistency.""",
                                """Welcome to our advanced text-to-speech system with voice cloning capabilities. 
This technology can handle documents, articles, and other long-form content. 
The parallel processing ensures that even lengthy texts are converted quickly. 
You can upload a reference audio sample to clone any voice you want. 
The same voice will be used consistently across all chunks for natural-sounding results."""
                            ],
                            inputs=long_text,
                            label="Example Long Texts"
                        )
                    
                    with gr.Column(scale=1):
                        long_audio = gr.Audio(label="Generated Long Form Speech")
                        long_result = gr.Textbox(label="Generation Stats", interactive=False)
                
                long_submit.click(
                    fn=generate_long_form_speech,
                    inputs=[model_state, long_text, long_ref_wav, long_exaggeration, long_temp, long_cfg_weight, long_seed_num, batch_size, max_chunk_size],
                    outputs=[long_audio, long_result]
                )
            
            # Tab 3: Utilities
            with gr.Tab("Utilities"):
                with gr.Column():
                    gr.Markdown("### File Management")
                    cleanup_btn = gr.Button("Clean Up Generated Files", variant="secondary")
                    cleanup_result = gr.Textbox(label="Cleanup Result", interactive=False)
                    
                    gr.Markdown("""
                    ### Tips for Best Results:
                    * **Reference Audio**: Use high-quality audio samples (clear, no background noise)
                    * **Duration**: 3-10 seconds of reference audio usually works best
                    * **Exaggeration**: 0.5 is neutral, higher values add more emotion
                    * **Temperature**: Lower values (0.1-0.8) for more consistent output
                    * **CFG/Pace**: Higher values for faster speech, lower for slower
                    * **Chunk Size**: Smaller chunks for better voice consistency in long form
                    """)
                    
                    cleanup_btn.click(
                        fn=cleanup_files_with_result,
                        outputs=[cleanup_result]
                    )
        
        # Load model on startup
        demo.load(
            fn=load_model, 
            inputs=[], 
            outputs=model_state
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_gradio_interface()
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(share=True)
