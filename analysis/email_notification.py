import smtplib
from getpass import getpass


class NotificationSystem():
    def __init__(self):
        self.be_notified = True
        self.email_address = 'tensu.wave@gmail.com'
        validation = input('Type `yes or y` if you wish to be notified by email:\t')
        if not (validation == 'yes' or validation == 'y'):
            self.be_notified = False
            return
        message = 'Email password for ' + self.email_address + ' :'
        self.password = getpass(message)

    def send_message(self, title, body):
        if not self.be_notified :
            return
        smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
        smtp_server.ehlo()
        smtp_server.starttls()
        smtp_server.login(self.email_address, self.password)
        content = 'Subject: ' + title + '\n' + body
        smtp_server.sendmail(self.email_address, 'tensu.wave@gmail.com', content)

        smtp_server.quit()
        print('Email sent successfully')
