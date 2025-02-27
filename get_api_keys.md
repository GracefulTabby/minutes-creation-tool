# Getting API Keys for Minutes Creation Tool

This guide will help you obtain the necessary API keys for the Minutes Creation Tool.

## AssemblyAI API Key

1. Go to [AssemblyAI's website](https://www.assemblyai.com/) and sign up for an account.
2. After signing up and logging in, navigate to your dashboard.
3. Your API key should be visible on the dashboard. If not, look for an "API" or "Settings" section.
4. Copy the API key and add it to your `.env` file as `ASSEMBLYAI_API_KEY=your_key_here`.

## Anthropic API Key

1. Go to [Anthropic's Console](https://console.anthropic.com/) and sign up for an account.
2. After signing up and logging in, navigate to the API Keys section.
3. Click on "Create Key" and give it a name (e.g., "Minutes Creation Tool").
4. Copy the API key (note that you won't be able to see it again after closing the page).
5. Add it to your `.env` file as `ANTHROPIC_API_KEY=your_key_here`.

## Updating Your .env File

Your `.env` file should look something like this:

```
# API Keys
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Directory Settings
AUDIO_DIR=audio_files
OUTPUT_DIR=output_files
```

Replace `your_assemblyai_api_key_here` and `your_anthropic_api_key_here` with the actual API keys you obtained.

## API Usage and Costs

Both AssemblyAI and Anthropic are paid services, but they offer free tiers or credits for new users:

- **AssemblyAI**: Offers a free tier with limited hours of audio processing per month.
- **Anthropic**: Offers some free credits for new users, after which you'll need to pay for usage.

Be sure to check the current pricing on their respective websites to understand potential costs based on your usage.
