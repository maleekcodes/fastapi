import replicate

def transcribe(audio_file: str):
    input = {
        "file_url": audio_file
    }

    output = replicate.run(
        "thomasmol/whisper-diarization:1495a9cddc83b2203b0d8d3516e38b80fd1572ebc4bc5700ac1da56a9b3ed886",
        input=input)

    return output

if __name__ == "__main__":
    print(transcribe("test.mp4"))