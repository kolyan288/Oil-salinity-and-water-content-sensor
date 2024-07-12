import asyncio
import requests
import argparse
from aiogram import Bot, Dispatcher

url = "https://api.telegram.org/bot7440138525:AAEuKdyrLFr0jsuW-IYzYhQ-dSAgnF7NHrM/getUpdates"

response = requests.get(url)

if response.status_code == 200:
    
    data = response.json()
    chat_ids = {update['message']['chat']['id'] for update in data['result']}
    chat_ids = list(chat_ids)

else:
    print(f"Ошибка при выполнении запроса: {response.status_code}")

parser = argparse.ArgumentParser(description="Process a list of integers.")
parser.add_argument("predicted_class")
parser.add_argument("feature_vector")
args = parser.parse_args()
print(args)
cls = eval(args.predicted_class)
fv = eval(args.feature_vector)

bot = Bot(token="7440138525:AAEuKdyrLFr0jsuW-IYzYhQ-dSAgnF7NHrM")
dp = Dispatcher()

@dp.message()
async def send_datetime(cls, fv, chat_ids):
    for chat_id in chat_ids: 
   #for chat_id in ['757806504']: 
        await bot.send_message(chat_id, f"{fv}")

        if cls == 0:
            await bot.send_message(chat_id, f"Предсказанный класс - 0. Полет нормальный")
        else:
            await bot.send_message(chat_id, f"Предсказанный класс - 1. Следует принять меры")

async def main():
    #asyncio.create_task(send_datetime(lst))
    #await dp.start_polling(bot)
    await send_datetime(cls, fv, chat_ids)
    await bot.session.close()

# Запуск асинхронного цикла событий
asyncio.run(main())