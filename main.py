import wx

from SpamDetectionModel.spam_detection_model import SpamDetectionModel


class SpamDetectorFrame(wx.Frame):
    def __init__(self, model: SpamDetectionModel):
        super().__init__(
            None,
            title="Spam Detector",
            size=wx.Size(700, 500),
            style=wx.MINIMIZE_BOX | wx.CAPTION | wx.CLOSE_BOX | wx.SYSTEM_MENU,
        )

        self.model = model

        panel = wx.Panel(self)
        panel.SetBackgroundColour(wx.Colour(245, 246, 250))

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # --- header ---
        header_sizer = wx.BoxSizer(wx.VERTICAL)

        title = wx.StaticText(panel, label="Spam Detector")
        title_font = title.GetFont()
        title_font.PointSize += 6
        title_font = title_font.Bold()
        title.SetFont(title_font)

        subtitle = wx.StaticText(
            panel, label="Paste a message below and check if it looks like spam."
        )
        subtitle.SetForegroundColour(wx.Colour(100, 100, 100))

        header_sizer.Add(title, 0, wx.BOTTOM, 4)
        header_sizer.Add(subtitle, 0, wx.BOTTOM, 10)

        main_sizer.Add(header_sizer, 0, wx.ALL | wx.EXPAND, 15)

        # --- Text input ---
        input_label = wx.StaticText(panel, label="Message")
        input_font = input_label.GetFont()
        input_font = input_font.Bold()
        input_label.SetFont(input_font)

        self.text_input = wx.TextCtrl(
            panel,
            style=wx.TE_MULTILINE | wx.TE_WORDWRAP | wx.BORDER_SIMPLE,
            size=wx.Size(-1, 200),
        )

        main_sizer.Add(input_label, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 15)
        main_sizer.Add(self.text_input, 1, wx.LEFT | wx.RIGHT | wx.EXPAND, 15)

        # --- Buttons ---
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.check_btn = wx.Button(panel, label="Check for spam")
        self.clear_btn = wx.Button(panel, label="Clear")
        self.sample_btn = wx.Button(panel, label="Sample message")

        btn_sizer.Add(self.check_btn, 0, wx.RIGHT, 10)
        btn_sizer.Add(self.clear_btn, 0, wx.RIGHT, 10)
        btn_sizer.Add(self.sample_btn, 0)

        main_sizer.Add(btn_sizer, 0, wx.ALL | wx.ALIGN_LEFT, 15)

        # --- Result box ---
        result_box = wx.StaticBox(panel, label="Result")
        result_sizer = wx.StaticBoxSizer(result_box, wx.VERTICAL)

        self.result_label = wx.StaticText(panel, label="No check yet")
        result_font = self.result_label.GetFont()
        result_font.PointSize += 3
        result_font = result_font.Bold()
        self.result_label.SetFont(result_font)
        self.result_label.SetForegroundColour(wx.Colour(120, 120, 120))

        self.result_desc = wx.StaticText(
            panel,
            label="Run a check to see if this message is likely spam.",
            style=wx.ST_NO_AUTORESIZE,
        )
        self.result_desc.Wrap(600)
        self.result_desc.SetForegroundColour(wx.Colour(90, 90, 90))

        result_sizer.Add(self.result_label, 0, wx.ALL, 5)
        result_sizer.Add(self.result_desc, 0, wx.ALL | wx.EXPAND, 5)

        main_sizer.Add(result_sizer, 0, wx.ALL | wx.EXPAND, 15)

        panel.SetSizer(main_sizer)

        # --- Status bar ---
        self.status_bar = self.CreateStatusBar()
        self.status_bar.SetStatusText("Ready to check")

        # --- Bind events ---
        self.check_btn.Bind(wx.EVT_BUTTON, self.on_check)
        self.clear_btn.Bind(wx.EVT_BUTTON, self.on_clear)
        self.sample_btn.Bind(wx.EVT_BUTTON, self.on_sample)

        self.Centre()
        self.Show()

    def on_check(self, event):
        text = self.text_input.GetValue().strip()

        if not text:
            wx.MessageBox(
                "Please enter a message to check.",
                "No text",
                wx.OK | wx.ICON_INFORMATION,
            )
            return

        try:
            prediction = self.model.predict(text)
        except Exception as e:
            wx.MessageBox(
                f"Error while running prediction:\n{e}",
                "Error",
                wx.OK | wx.ICON_ERROR,
            )
            self.status_bar.SetStatusText("Error during prediction")
            return

        is_spam = prediction == 1
        self.update_result_ui(is_spam)
        self.status_bar.SetStatusText(
            "Checked: spam" if is_spam else "Checked: not spam"
        )

    def on_clear(self, event):
        self.text_input.SetValue("")
        self.result_label.SetLabel("No check yet")
        self.result_label.SetForegroundColour(wx.Colour(120, 120, 120))
        self.result_desc.SetLabel("Run a check to see if this message is likely spam.")
        self.status_bar.SetStatusText("Cleared")

    def on_sample(self, event):
        sample = (
            "Congratulations! You have WON a free cruise! "
            "Call 923423333 now to claim your FREE prize. Limited time offer!"
        )
        self.text_input.SetValue(sample)
        self.text_input.SetInsertionPointEnd()
        self.status_bar.SetStatusText("Inserted sample message")

    def update_result_ui(self, is_spam: bool):
        if is_spam:
            self.result_label.SetLabel("SPAM")
            self.result_label.SetForegroundColour(wx.Colour(200, 0, 0))
            self.result_desc.SetLabel(
                "This message looks like spam. Be careful with links, attachments, "
                "and requests for personal information."
            )
        else:
            self.result_label.SetLabel("NOT SPAM")
            self.result_label.SetForegroundColour(wx.Colour(0, 150, 0))
            self.result_desc.SetLabel(
                "This message does not look like spam. Still, stay cautious with "
                "unknown senders."
            )


def main():
    sdm = SpamDetectionModel("./dataset/SMSSpamCollection")

    app = wx.App(False)
    SpamDetectorFrame(sdm)
    app.MainLoop()


if __name__ == "__main__":
    main()
