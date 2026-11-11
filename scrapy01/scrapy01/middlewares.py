from selenium.webdriver import Chrome
from selenium.webdriver.chrome.service import Service
from scrapy.http import HtmlResponse

class SeleniumMiddleware:
    def process_request(self,requse,spider):
        s = Service(executable_path="selenium/chromedriver.exe")
        chrome = Chrome(service=s)
        chrome.get('https://httpbin.org/get')

        return HtmlResponse(
            url=requse.url,
            body=chrome.page_source,
            request=requse,
            encoding='utf-8'
            )
