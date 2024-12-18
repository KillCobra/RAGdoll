import pyttsx3

class TextToSpeech:
    def __init__(self):
        # Initialize the text-to-speech engine
        self.engine = pyttsx3.init()
        print("Text-to-Speech system initialized")

    def text_to_speech(self, text):
        """Convert text to speech using pyttsx3"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            raise RuntimeError(f"Failed to generate/play speech: {str(e)}")

def main():
    # Initialize TTS
    tts = TextToSpeech()

    # Example usage
    print("\nText-to-Speech Assistant")
    print("------------------------")
    print("Enter text to speak, or 'q' to quit")
    
    while True:
        try:
            # Get input text
            text = input("\nEnter text: ")
            if text.lower() == 'q':
                break

            # Generate and play speech
            print("Speaking...")
            tts.text_to_speech(text)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()