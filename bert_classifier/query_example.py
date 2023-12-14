import requests
import pandas as pd

# An example to test that the server works correctly.
# It takes one sample for each Iris type, requests prediction and compares it with the right target
if __name__ == '__main__':

df = pd.DataFrame(data = {'id_survey' : ['05dd0c57-2c40-4954-ba44-d23ed9ccb2a4'
                                    , '75072f49-a0ea-4d8d-8eca-16995d765cec'
                                    , 'b455f3d0-3986-4644-beca-9a482f89ef0c']
                     , 'text' : ['Глючил сайт, жал на оплату пошлины 5 раз, при это выкидывало на вопрос-ответ'
                                 , 'Постоянно получаю отказы от Мосжилинспекции. иногда даже непонятно что не так'
                                 , 'Консультацию специалиста введите и образец заполнения.']})

for comments in range(len(df)):
    resp = requests.post(
        "http://127.0.0.1:80/predict",
        json = {'text' : str(df.loc[comments]['text'])}
        )
    

        print(f"Input features: {str(df.loc[comments]['text'])}")
        print(f"Predicted: {resp.json()}")
	print(f"id_survey: {str(df.loc[comments]['id_survey'])}")
        print("----")
