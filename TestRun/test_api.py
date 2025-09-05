#!/usr/bin/env python3
"""
Test script to verify OpenAI API connection
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

def test_openai_api():
    """Test OpenAI API connection and model availability"""
    try:
        # Initialize client
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        print("🔑 API Key loaded from environment")
        
        # Test with a simple message
        print("📡 Testing API connection...")
        
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Respond briefly."},
                {"role": "user", "content": "Say hello in one sentence."}
            ],
            max_completion_tokens=50,
            temperature=0.7
        )
        
        # Extract response
        reply = response.choices[0].message.content
        
        print("✅ API Connection Successful!")
        print(f"🤖 Model Response: {reply}")
        print(f"📊 Tokens Used: {response.usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ API Test Failed: {str(e)}")
        
        # Check common issues
        if "api_key" in str(e).lower():
            print("💡 Issue: Check your OPENAI_API_KEY in .env file")
        elif "model" in str(e).lower():
            print("💡 Issue: gpt-5-nano model may not be available")
            print("   Try: gpt-4o-mini or gpt-3.5-turbo instead")
        elif "quota" in str(e).lower():
            print("💡 Issue: API quota exceeded or payment required")
        
        return False

if __name__ == "__main__":
    print("🧪 OpenAI API Test")
    print("=" * 30)
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("📝 Create .env file with: OPENAI_API_KEY=your_key_here")
        exit(1)
    
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("❌ OPENAI_API_KEY not set in .env file!")
        print("📝 Edit .env file and add your real API key")
        exit(1)
    
    # Run test
    success = test_openai_api()
    
    if success:
        print("\n🎉 Ready to run the WeChat bot!")
    else:
        print("\n🔧 Fix the issues above before running the bot")