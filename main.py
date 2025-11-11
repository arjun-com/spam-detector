from pprint import pprint
from SpamDetectionModel.spam_detection_model import SpamDetectionModel

sdm = SpamDetectionModel("./dataset/SMSSpamCollection")

pprint(sdm.predict("Call 923423333 for a free burger, with no extra tax.")[0])
pprint(sdm.predict("Should we goto John's house tmrw?")[0])
