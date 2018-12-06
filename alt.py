"""
 Arbitrarily opening commands from voice activation
"""

import os
import subprocess
import sys
import pyaudio, pyttsx3
import speech_recognition as sr


KEYWORDS = ["excel", "internet", "music", "hibernate", "firefox"]
QUIT = ["stop", "David", "listening", "you"]
INTRO = "Hello and welcome, I am David"


def say(text):
    """responds to the query using a text to speech module"""
    engine = pyttsx3.init()
    engine.setProperty('voice', 'english')  # changes the voice
    engine.say(text)
    engine.runAndWait()

def close(component):
    """closes the component opened by the system"""
    if component == "David":
        sys.exit(0)
    else:
        pass

def open_this(component):
    """opens a component"""
    try:
        print(component)
        subprocess.Popen(component)
    except:
        say("command was not recognized")
    return True

def controller(source):
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)
    try:
        user = r.recognize_google(audio, language="en-us")
        sentence = user.lower()
        sent_list = sentence.split()
        print(sentence)
        if len(set(QUIT).intersection(set(sent_list))) >=2 :
            say("stopping David")
            close("David")
        if "david please open" in sentence or "david open" in sentence:
            component = sent_list.index("open") + 1
            open_this(sent_list[component:])
        if "david please close" in sentence or "david close" in sentence:
            component = sent_list.index("close") + 1
            close(sent_list[component:])
    except sr.RequestError:
        user = r.recognize_sphinx(audio, language="en-US")
        sentence = user.lower()
        sent_list = sentence.split()
        if len(set(QUIT).intersection(set(sent_list))) >=2 :
            say("stopping David")
            close("David")
    except sr.UnknownValueError:
        print("Could not understand statement")

if __name__ == "__main__":
    say(INTRO)
    r = sr.Recognizer()
    with sr.Microphone() as source:
        while 1:
            controller(source)

