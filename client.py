import requests
import json

def query_granite(user_prompt, system_prompt="You are a math reasoning assistant.", context="", api_url="https://nab6wk9x0oev1u-8888.proxy.runpod.net/api/granite/generate", timeout=300):

    payload = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "context": context
    }

    try:
        response = requests.post(
            api_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=timeout
        )

        print("Status Code:", response.status_code)
        print("Raw Response (first 500 chars):", response.text[:500])

        if response.status_code == 200:
            result = response.json()
            return result.get("output", "No output found in response.")
        else:
            return "❌ Non-200 response, cannot parse JSON. Check URL and route."

    except requests.exceptions.Timeout:
        return "❌ Request timed out. Model may still be loading."
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to the server. Check the proxy URL."
    except Exception as e:
        return f"❌ Unexpected error: {e}"

# Example usage
if __name__ == "__main__":
    output = query_granite("Compute the derivative of x^3 * ln(x).")
    print("\n🧠 Granite Model Output:\n", output)
