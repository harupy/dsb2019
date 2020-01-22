import re
import requests

from credentials import credentials


def parse_reset_date(html):
    """
    Parse submissions reset date from HTML.
    """
    m = re.search(r'"maxDailySubmissionsResetDate":"(.+?)"', html)
    return m.group(1)


def parse_remaining_submissions(html):
    """
    Parse remaining submissions from HTML.
    """
    m = re.search(r'"remainingDailySubmissions":(\d+)', html)
    return int(m.group(1))


def get_authorized_session():
    """
    Create an authorized session.
    """
    # create a session
    session = requests.Session()

    # sign in
    SIGNIN_URL = 'https://www.kaggle.com/account/email-signin'
    session.get(SIGNIN_URL)  # visit the sign-in page to get a cooky.
    data = {
        'email': credentials['username'],
        'password': credentials['password'],
        'X-XSRF-TOKEN': session.cookies['XSRF-TOKEN'],
    }
    session.post(SIGNIN_URL, data=data)
    return session


def main():
    session = get_authorized_session()
    resp = session.get('https://www.kaggle.com/c/data-science-bowl-2019/submit')
    print('Submissions Remaining', parse_remaining_submissions(resp.text))
    print('Submissions Reset Date', parse_reset_date(resp.text))


if __name__ == "__main__":
    main()
