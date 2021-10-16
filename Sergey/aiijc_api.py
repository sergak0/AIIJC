import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
import threading
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from flask import Flask, Response
import flask
import time
from flask_restplus import Api, Resource, fields
# import flask.scaffold
# flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func

flask_app = Flask(__name__)
api = Api(app=flask_app,
          title="Api for changing text topic",
          description="Developed by Dungeon Masters command")

chrome_options = ChromeOptions()
chrome_options.add_argument("--headless")
driver = webdriver.Remote(
    command_executor='http://0.0.0.0:4444/wd/hub',
    options=chrome_options,
)

translate_variants = ["sport", "music", "literature", "animals"]
language_variants = ["en", "ru"]

request_model = api.model('Model',
                          {'text': fields.String(required=True,
                                                 description="The text, which you want to convert in other topic"),
                           'topic': fields.String(required=True,
                                                  description='The topic you want to translate the text to.\n'
                                                              f'One of {str(translate_variants)}'),
                           'language': fields.String(required=False,
                                                     description="The language in which the text is written.\n"
                                                                 f'One of {str(language_variants)}')
                           })


@api.route('/convert_text', methods=['POST'])
class DetectedExercise(Resource):
    @api.doc(responses={200: 'OK', 404: 'Bad json'})
    @api.expect(request_model)
    def post(self):
        data = flask.request.json

        if data['topic'] not in translate_variants:
            return Response(f'Topic must be one of {translate_variants}', status=404)

        if data['language'] not in language_variants:
            return Response(f'Topic must be one of {language_variants}', status=404)

        if data['language'] == 'en':
            data['text'] = translate_text(data['text'], 'en-ru')

        data['predict'] = 'Новый текст'

        if data['language'] == 'en':
            data['predict'] = translate_text(data['predict'], 'ru-en')

        print(data['predict'])
        return {'text': data['predict']}


def translate_text(text, direction='ru-en'):
    time.sleep(0.2)
    driver.get(f"https://www.m-translate.ru/translator/text#text=test&direction=ru-en")

    from_lang = driver.find_element_by_id("from_span").text.lower()
    if from_lang != direction[:direction.find('-')]:
        driver.find_element_by_id("Small_btns").click()

    driver.find_element_by_id("text").clear()
    driver.find_element_by_id("text").send_keys(text)
    driver.find_element_by_id("go_btn").click()
    while driver.find_element_by_id("text_out").get_attribute("value") == "":
        time.sleep(0.2)
    return driver.find_element_by_id("text_out").get_attribute("value")


if __name__ == '__main__':
    pass
    # threading.Thread(target=flask_app.run, kwargs={'host': '0.0.0.0', 'port': '1212'}).start()
