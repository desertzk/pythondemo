# Download the helper library from https://www.twilio.com/docs/python/install
from twilio.rest import Client


# Your Account Sid and Auth Token from twilio.com/console
# DANGER! This is insecure. See http://twil.io/secure
account_sid = 'AC0b49849d19ba300477fd164bd681e88c'
auth_token = '8b37874e6cac1f01e3a7b6f1d1e14ada'
client = Client(account_sid, auth_token)

message = client.messages .create(
                     body="7777777777777777",
                     from_='+16788206970',
                     to='+8618758208569'
                 )

print(message.sid)