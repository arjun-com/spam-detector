from pprint import pprint
from SpamDetectionModel.spam_detection_model import SpamDetectionModel
import wx

sdm = SpamDetectionModel("./dataset/SMSSpamCollection")

pprint(sdm.predict("Call 923423333 for a free burger, with no extra tax."))
pprint(sdm.predict("Should we goto John's house tmrw?"))

app = wx.App(False)

frame = wx.Frame(None, title = "Spam Detector", size = wx.Size(800, 600), style = wx.MINIMIZE_BOX | wx.CAPTION | wx.CLOSE_BOX)

sizer = wx.BoxSizer(wx.VERTICAL)

textInput = wx.TextCtrl(frame, style = wx.TE_MULTILINE, size = wx.Size(500, 300))
sizer.Add(textInput, 0, 0, 0)

submitBtn = wx.Button(frame, label = "Check for spam")
sizer.Add(submitBtn, 0, 0, 0)

def handleSubmitBtn(_):
    text = textInput.GetValue()
    res = sdm.predict(text) == 1

    wx.MessageBox(f"This email is most likely {'spam' if res else 'not spam'}", "Spam Check Results")

submitBtn.Bind(wx.EVT_BUTTON, handleSubmitBtn)

frame.SetSizer(sizer)

frame.Show()

app.MainLoop()
