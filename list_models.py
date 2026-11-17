from nlu.llm_client import get_gemini_client


def main():
    try:
        with get_gemini_client() as client:
            for model in client.models.list():
                print(model.name)
    except Exception:
        print("Unable to list models. Check API configuration and connection.")


if __name__ == "__main__":
    main()
