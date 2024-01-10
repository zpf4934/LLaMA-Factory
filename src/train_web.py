from llmtuner import create_ui


def main():
    demo = create_ui()
    demo.queue(concurrency_count=10)
    demo.launch(server_name="0.0.0.0", share=False, inbrowser=True)


if __name__ == "__main__":
    main()
