# Visual-Hand-sign-detector-
Welcome to my sign language interpreter project. I built this to help bridge the communication gap between sign language users and those who don't know ASL. It's pretty cool, just show your hand to the webcam, and it translates your gestures into text in real time. 
What Does It Do? So basically, this app watches your hand through your webcam and recognizes ASL (American Sign Language) letters and numbers. When you make a sign and hold it for about 1.5 seconds, it adds that letter to the screen. It's like typing, but with your hands.
I built this using Python, OpenCV for handling the camera, and Google's MediaPipe for tracking hand movements. It can recognize about 20 different letters and numbers 1-5 right now.
I've always been interested in accessibility tech, and I wanted to create something that could actually help people. Plus, it's a great way to learn computer vision and machine learning. While there are similar projects out there, I wanted to build my own from scratch to really understand how it works.
The idea is simple: if someone knows sign language but the person they're talking to doesn't, this could help translate basic messages. It's not perfect (more on that later), but it's a start!
Cool Features
Here's what the app can do:

Real-time tracking - Your hand movements are tracked instantly at around 30 frames per second

Visual feedback - You can see exactly where the app thinks your hand landmarks are (those little dots and lines on your hand)

Smart recognition - Instead of registering every tiny movement, it waits for you to hold a gesture for 1.5 seconds to confirm

Builds sentences - As you sign different letters, they get added together into words

Works with both hands - Though honestly, it works best with one hand at a time

Getting Started
What You'll Need
Before you start, make sure you have:

Python 3.8 or newer installed
A working webcam (built-in or external)
A decent internet connection for downloading packages
About 5 minutes to set everything up
What It Can Recognize
Right now, the app recognizes these signs:
Letters (17 of them)

A - Make a fist with your thumb on the side

B - Flat hand, all fingers together pointing up

C - Curve your hand like you're holding a cup

D - Point your index finger up, other fingers touch your thumb

E - Curl all your fingers in tight

F - Make an "OK" sign but keep your other fingers pointing up

G - Point your index finger and thumb to the side

H - Index and middle finger pointing sideways together

I - Just stick your pinky up

K - Index finger up, middle finger at an angle

L - Make an L shape with your thumb and index finger

O - Make a circle with all your fingers

R - Cross your index and middle fingers

U - Index and middle fingers together pointing up

V - Peace sign (but fingers close together)

W - Three fingers up

Y - Thumb and pinky out (hang loose!)

What's Missing?
Some letters are tricky because they involve movement (like J and Z) or are really similar to others (like M and N). I'm working on adding those.
Tips for Better Results
After testing this a bunch, here's what I learned works best:
Lighting matters!

Use bright lighting, but not from behind you
Avoid harsh shadows on your hand
Natural daylight works great

Camera position

Keep your hand about arm's length from the camera
Try to keep your palm facing the camera
Make sure your whole hand is visible in the frame

Making gestures

Be deliberate with your hand shapes
Hold each sign really still for the full 1.5 seconds
Don't rush between letters
Practice a bit - some signs take getting used to!

Environment

A plain background helps (like a white wall)
Avoid busy patterns behind you
Make sure good lighting is on your hand, not behind it.
How It Actually Works
This is the fun part! Here's what's happening behind the scenes:

Video capture - OpenCV grabs frames from your webcam

Hand detection - MediaPipe finds your hand in each frame and identifies 21 key points (landmarks) on it

Analysis - The app looks at where these points are and calculates things like:

Which fingers are extended
Distances between different points
Overall hand shape


Pattern matching - It compares what it sees to known ASL signs

Confirmation - If the same sign is detected for 1.5 seconds, it's added to the text

Display - Everything gets drawn on screen with those cool skeleton overlays

The reason for the 1.5 second hold is to avoid false positives. Without it, the app would register random movements as letters, which gets messy fast!
Current Limitations
Let me be honest about what this can't do yet:

Limited vocabulary - Only 17 letters and 5 numbers for now

No moving signs - Signs that need motion (like J or Z) don't work yet

Some similar signs - Letters like M, N, S, and T are hard to tell apart

One hand at a time - While it can see two hands, recognition works best with one

Needs good conditions - Lighting and background really matter

Not super fast - That 1.5 second hold can feel slow when you're spelling out words

This is very much a work in progress.
