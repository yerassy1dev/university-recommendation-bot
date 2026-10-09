# University Recommendation Bot

An AI-powered Telegram bot that helps students discover which Hong Kong university programmes best match their interests, strengths, career goals, location, and budget, using a weighted scoring algorithm and Google Gemini.

## Features

- **Chat with AI:** Describe yourself in your own words, and Gemini turns your message into a structured student profile.
- **Guided Questions:** Prefer buttons? Answer five quick questions about interests, strong subjects, career goal, location, and budget.
- **Programme Matching:** Calculate a match score for each programme using a weighted algorithm and show the top five.
- **Personalised Explanations:** Get a short AI-written explanation of why each programme fits you, with a template fallback if the AI is unavailable.
- **Interactive Follow-up Chat:** Ask questions about your results, compare programmes, or say "what if my budget is lower?" to re-run the matching.
- **Data Persistence:** Store each search and its results in an SQLite database and view the latest with `/last`.

## How the Match Score Works

The ranking is deterministic, so scores stay predictable and explainable. Gemini is used only for the language parts (understanding text, explanations, and chat).

| Factor | Weight |
| --- | --- |
| Academic interests | 30% |
| Career goal | 25% |
| Subject strengths | 20% |
| Budget | 15% |
| Location | 10% |

These weights are initial design choices, not scientifically validated measures. They can be adjusted in the `WEIGHTS` dictionary.

## Tech Stack

- **Language:** Python
- **Bot Framework:** Aiogram 3
- **AI Integration:** Google Gemini API
- **Data Processing:** Pandas
- **Database:** SQLite
- **Configuration:** python-dotenv

## Project Structure

```text
UniversityRecommendationBot/
├── university_bot.py
├── programmes.csv
├── README.md
├── .gitignore
└── .env.example
```

The `.env` file and SQLite database are created or stored locally and should not be committed to GitHub.

## Dataset

`programmes.csv` currently contains **sample data** for 18 Hong Kong programmes. Tuition figures and admission notes are placeholders and must be verified against official university websites before the bot is used for real recommendations. To add a programme, add a new row and separate multiple values in a cell with `;` (for example `AI;Programming;Robotics`).

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/university-recommendation-bot.git
cd university-recommendation-bot
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Install dependencies

```bash
python -m pip install aiogram pandas python-dotenv google-genai
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```dotenv
BOT_TOKEN=your_telegram_bot_token
GEMINI_API_KEY=your_gemini_api_key
```

Replace the placeholders with your own credentials.

- Create a Telegram bot using [BotFather](https://t.me/BotFather).
- Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

### 4. Run the bot

```bash
python university_bot.py
```

Open your Telegram bot and send `/start` to begin.

## Commands

| Command | Description |
| --- | --- |
| `/start` | Begin a new search (chat with AI or guided questions) |
| `/last` | Show your most recent results |
| `/help` | Show help and the score weights |

## Security

- Never commit your `.env` file or expose API keys.
- Keep credentials in environment variables.
- Do not upload personal databases or private user data.

## Limitations

- Match scores are illustrative and are not admission probabilities.
- Programme data is sample data, and AI-generated text can contain mistakes. Always check fees and requirements on the official university website.

## Future Improvements

- Replace the sample data with a verified dataset of 30-50 programmes from official sources.
- Use embeddings for semantic matching of interests and careers.
- Add more locations and fields of study.
- Improve error handling and API usage limits.

## Author

Developed as a personal software project to explore Python, AI integration, Telegram bot development, and recommendation systems.