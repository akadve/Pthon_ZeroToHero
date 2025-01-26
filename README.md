# Python_ZeroToHero

# Python Learning Course: Gamified Edition

## Table of Contents
1. [Introduction](#introduction)
2. [Module 1: Basics of Python](#module-1-basics-of-python)
    - [Lesson 1: Introduction to Python](#lesson-1-introduction-to-python)
    - [Lesson 2: Variables and Data Types](#lesson-2-variables-and-data-types)
    - [Lesson 3: Basic Operators](#lesson-3-basic-operators)
    - [Game Example: Number Guessing Game](#game-example-number-guessing-game)
3. [Module 2: Control Structures](#module-2-control-structures)
    - [Lesson 1: Conditional Statements](#lesson-1-conditional-statements)
    - [Lesson 2: Loops](#lesson-2-loops)
    - [Game Example: Rock, Paper, Scissors](#game-example-rock-paper-scissors)
4. [Module 3: Functions and Modules](#module-3-functions-and-modules)
    - [Lesson 1: Defining Functions](#lesson-1-defining-functions)
    - [Lesson 2: Importing Modules](#lesson-2-importing-modules)
    - [Game Example: Hangman](#game-example-hangman)
5. [Module 4: Data Structures](#module-4-data-structures)
    - [Lesson 1: Lists](#lesson-1-lists)
    - [Lesson 2: Tuples](#lesson-2-tuples)
    - [Lesson 3: Dictionaries](#lesson-3-dictionaries)
    - [Lesson 4: Sets](#lesson-4-sets)
    - [Game Example: Word Jumble](#game-example-word-jumble)
6. [Module 5: File Handling](#module-5-file-handling)
    - [Lesson 1: Reading and Writing Files](#lesson-1-reading-and-writing-files)
    - [Game Example: Quiz Game](#game-example-quiz-game)
7. [Module 6: Object-Oriented Programming (OOP)](#module-6-object-oriented-programming-oop)
    - [Lesson 1: Classes and Objects](#lesson-1-classes-and-objects)
    - [Lesson 2: Inheritance and Polymorphism](#lesson-2-inheritance-and-polymorphism)
    - [Game Example: Simple RPG](#game-example-simple-rpg)
8. [Module 7: Advanced Topics](#module-7-advanced-topics)
    - [Lesson 1: Error Handling](#lesson-1-error-handling)
    - [Lesson 2: Regular Expressions](#lesson-2-regular-expressions)
    - [Lesson 3: List Comprehensions](#lesson-3-list-comprehensions)
    - [Game Example: Text Adventure](#game-example-text-adventure)
9. [Module 8: Data Science](#module-8-data-science)
    - [Lesson 1: NumPy Arrays](#lesson-1-numpy-arrays)
    - [Lesson 2: Matplotlib Diagrams](#lesson-2-matplotlib-diagrams)
    - [Lesson 3: Pandas Data Analysis](#lesson-3-pandas-data-analysis)
    - [Game Example: Analyze Your Friends' Ages](#game-example-analyze-your-friends-ages)
10. [Module 9: Machine Learning](#module-9-machine-learning)
    - [Lesson 1: Linear Regression](#lesson-1-linear-regression)
    - [Lesson 2: Classification Algorithms](#lesson-2-classification-algorithms)
    - [Lesson 3: Neural Networks](#lesson-3-neural-networks)
    - [Game Example: Predict House Prices](#game-example-predict-house-prices)
11. [Module 10: Finance](#module-10-finance)
    - [Lesson 1: Loading Financial Data](#lesson-1-loading-financial-data)
    - [Lesson 2: Graphical Visualization](#lesson-2-graphical-visualization)
    - [Lesson 3: Trendlines](#lesson-3-trendlines)
    - [Game Example: Stock Market Simulation](#game-example-stock-market-simulation)
12. [Module 11: Computer Vision](#module-11-computer-vision)
    - [Lesson 1: Loading Images and Videos](#lesson-1-loading-images-and-videos)
    - [Lesson 2: Thresholding](#lesson-2-thresholding)
    - [Lesson 3: Filtering](#lesson-3-filtering)
    - [Game Example: Find Hidden Objects](#game-example-find-hidden-objects)
13. [Conclusion](#conclusion)

---

## Introduction
Welcome to the gamified Python learning course! This course is designed for individuals of all ages who want to learn Python from scratch to advanced levels. Each module includes lessons with detailed explanations followed by a game example that you can execute to reinforce your learning.

If you think that this book has brought value to you and helped you on your programming journey, I would appreciate a quick review on Amazon. Thank you!

---

## Module 1: Basics of Python

### Lesson 1: Introduction to Python
```python
print("Hello, World!")
```
**Explanation:** The `print` function outputs text to the console. It's one of the first things you'll use in Python. Try typing different messages and see them appear on your screen!

### Lesson 2: Variables and Data Types
```python
name = "Alice"  # String
age = 30        # Integer
height = 5.5    # Float
is_student = True  # Boolean

print(name, age, height, is_student)
```
**Explanation:** Variables store data. Python supports several data types like strings, integers, floats, and booleans. Play around with these variables by changing their values and printing them.

### Lesson 3: Basic Operators
```python
a = 10
b = 3
print(a + b)  # Addition
print(a - b)  # Subtraction
print(a * b)  # Multiplication
print(a / b)  # Division
print(a % b)  # Modulus
print(a ** b) # Exponentiation
print(a // b) # Floor Division
```
**Explanation:** Operators perform operations on variables and values. Common arithmetic operators include addition, subtraction, multiplication, division, modulus, exponentiation, and floor division. Experiment with different numbers and see what happens!

### Game Example: Number Guessing Game
```python
import random

def number_guessing_game():
    number_to_guess = random.randint(1, 100)
    attempts = 0
    while True:
        guess = int(input("Guess a number between 1 and 100: "))
        attempts += 1
        if guess < number_to_guess:
            print("Too low!")
        elif guess > number_to_guess:
            print("Too high!")
        else:
            print(f"Congratulations! You guessed it in {attempts} attempts.")
            break

number_guessing_game()
```

---

## Module 2: Control Structures

### Lesson 1: Conditional Statements
```python
x = 10
if x > 5:
    print("x is greater than 5")
elif x == 5:
    print("x is equal to 5")
else:
    print("x is less than 5")
```
**Explanation:** Conditional statements allow you to execute code based on certain conditions. If the condition is true, the block of code inside the if statement will be executed. Try changing the value of `x` and see how the output changes.

### Lesson 2: Loops
```python
for i in range(5):
    print(i)

while count < 5:
    print(count)
    count += 1
```
**Explanation:** Loops allow you to repeat a block of code multiple times. Python supports both `for` and `while` loops. Try using different ranges and see what happens!

### Game Example: Rock, Paper, Scissors
```python
import random

def rock_paper_scissors():
    choices = ['rock', 'paper', 'scissors']
    computer_choice = random.choice(choices)
    player_choice = input("Enter rock, paper, or scissors: ").lower()

    print(f"Computer chose: {computer_choice}")
    if (player_choice == 'rock' and computer_choice == 'scissors') or \
       (player_choice == 'scissors' and computer_choice == 'paper') or \
       (player_choice == 'paper' and computer_choice == 'rock'):
        print("You win!")
    elif player_choice == computer_choice:
        print("It's a tie!")
    else:
        print("You lose!")

rock_paper_scissors()
```

---

## Module 3: Functions and Modules

### Lesson 1: Defining Functions
```python
def greet(name):
    return f"Hello, {name}!"

print(greet("Bob"))
```
**Explanation:** Functions are reusable blocks of code that perform specific tasks. They help organize code and make it more readable. Define your own functions and play around with them!

### Lesson 2: Importing Modules
```python
import math

print(math.sqrt(16))
```
**Explanation:** Modules contain additional functions and classes. Importing them allows you to use those functionalities. Try importing other modules like `random` and see what functions they offer.

### Game Example: Hangman
```python
import random

def hangman():
    words = ["python", "programming", "hangman"]
    word = random.choice(words)
    guesses = ''
    turns = 6

    while turns > 0:
        failed = 0
        for char in word:
            if char in guesses:
                print(char, end=' ')
            else:
                print("_", end=' ')
                failed += 1
        if failed == 0:
            print("\nYou won!")
            break
        guess = input("\nGuess a character: ")
        guesses += guess
        if guess not in word:
            turns -= 1
            print(f"Wrong! You have {turns} turns left")

    if turns == 0:
        print(f"You lose! The word was {word}")

hangman()
```

---

## Module 4: Data Structures

### Lesson 1: Lists
```python
fruits = ["apple", "banana", "cherry"]

print(fruits[0])  # Access elements
fruits.append("orange")  # Add element
print(fruits)
```
**Explanation:** Lists are ordered collections of items. You can access, modify, and add elements to lists. Create your own lists and experiment with them!

### Lesson 2: Tuples
```python
coordinates = (10, 20)

print(coordinates[0])
```
**Explanation:** Tuples are similar to lists but are immutable. Once created, their values cannot be changed. Use tuples when you need fixed data.

### Lesson 3: Dictionaries
```python
person = {"name": "John", "age": 30}

print(person["name"])
person["age"] = 31  # Update value
print(person)
```
**Explanation:** Dictionaries store key-value pairs. Keys must be unique, and you can easily update values. Create dictionaries about your friends and see what you can do with them!

### Lesson 4: Sets
```python
unique_numbers = {1, 2, 3, 4, 5}
unique_numbers.add(6)
print(unique_numbers)
```
**Explanation:** Sets contain unique elements. They support operations like union, intersection, and difference. Use sets when you need only unique values.

### Game Example: Word Jumble
```python
import random

def word_jumble():
    words = ["python", "jumble", "easy", "difficult", "answer"]
    word = random.choice(words)
    jumbled = ''.join(random.sample(word, len(word)))
    print(f"The jumbled word is: {jumbled}")

    guess = input("Your guess: ")
    if guess == word:
        print("Correct!")
    else:
        print(f"Incorrect. The word was {word}")

word_jumble()
```

---

## Module 5: File Handling

### Lesson 1: Reading and Writing Files
```python
with open('example.txt', 'w') as file:
    file.write('Hello, World!')

with open('example.txt', 'r') as file:
    content = file.read()
    print(content)
```
**Explanation:** File handling allows you to read from and write into files. Use `with` statements for better readability and to ensure files are closed properly.

### Game Example: Quiz Game
```python
questions = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is 2 + 2?", "answer": "4"},
]

score = 0
for q in questions:
    answer = input(q["question"] + ": ").lower()
    if answer == q["answer"].lower():
        score += 1

print(f"You got {score}/{len(questions)} correct!")
```

---

## Module 6: Object-Oriented Programming (OOP)

### Lesson 1: Classes and Objects
```python
class Dog:
    def __init__(self, name, breed):
        self.name = name
        self.breed = breed

    def bark(self):
        print(f"{self.name} says woof!")

dog1 = Dog("Buddy", "Golden Retriever")
dog1.bark()
```
**Explanation:** OOP is a programming paradigm centered around objects and classes. Classes define blueprints for objects. Create different animals and let them speak!

### Lesson 2: Inheritance and Polymorphism
```python
class Animal:
    def __init__(self, name):
        self.name = name

    def make_sound(self):
        print("Some sound!")

class Dog(Animal):
    def make_sound(self):
        print("Bark!")

dog = Dog("Buddy")
dog.make_sound()
```
**Explanation:** Inheritance allows one class to inherit attributes and methods from another. Override methods to change their behavior. Try creating other animals and making them speak!

### Game Example: Simple RPG
```python
class Character:
    def __init__(self, name, health):
        self.name = name
        self.health = health

    def attack(self, other):
        print(f"{self.name} attacks {other.name}")
        other.health -= 10

player = Character("Hero", 100)
enemy = Character("Monster", 50)
player.attack(enemy)
print(f"{enemy.name}'s health: {enemy.health}")
```

---

## Module 7: Advanced Topics

### Lesson 1: Error Handling
```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")
finally:
    print("This always executes")
```
**Explanation:** Error handling allows you to catch and handle exceptions gracefully. Try dividing by zero and see what happens!

### Lesson 2: Regular Expressions
```python
import re

text = "The rain in Spain"
match = re.search(r"ain", text)
if match:
    print("Found:", match.group())
```
**Explanation:** Regular expressions allow you to search and manipulate strings using patterns. Try finding different patterns in texts!

### Lesson 3: List Comprehensions
```python
squares = [x**2 for x in range(10)]
print(squares)
```
**Explanation:** List comprehensions provide a concise way to create lists. Try creating lists of cubes or other mathematical sequences.

### Game Example: Text Adventure
```python
def text_adventure():
    location = "forest"
    print("You are in a dark forest.")
    while True:
        command = input("Enter command: ").lower()
        if command == "look":
            print("You see trees and shadows.")
        elif command == "go north":
            print("You walk deeper into the forest.")
        elif command == "quit":
            print("Goodbye!")
            break
        else:
            print("Unknown command.")

text_adventure()
```

---

## Module 8: Data Science

### Lesson 1: NumPy Arrays
```python
import numpy as np

a = np.array([10, 20, 30])
print(a[0])

a = np.full((3, 3), 7)
print(a)
```
**Explanation:** NumPy arrays are efficient for numerical computations. Create multi-dimensional arrays and fill them with values.

### Lesson 2: Matplotlib Diagrams
```python
import matplotlib.pyplot as plt

x_values = np.linspace(0, 20, 100)
y_values = np.sin(x_values)

plt.plot(x_values, y_values)
plt.title("Sine Wave")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```
**Explanation:** Matplotlib helps visualize data. Plot different mathematical functions and see how they look.

### Lesson 3: Pandas Data Analysis
```python
import pandas as pd

data = {'Name': ['Anna', 'Bob', 'Charles'], 'Age': [24, 32, 35]}
df = pd.DataFrame(data)
print(df['Age'][1])
```
**Explanation:** Pandas provides powerful tools for data manipulation and analysis. Work with data frames and extract useful information.

### Game Example: Analyze Your Friends' Ages
```python
import pandas as pd

friends_data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 22]
}
df = pd.DataFrame(friends_data)

print("Average age:", df['Age'].mean())
print("Youngest friend:", df.loc[df['Age'].idxmin()]['Name'])
print("Oldest friend:", df.loc[df['Age'].idxmax()]['Name'])
```

---

## Module 9: Machine Learning

### Lesson 1: Linear Regression
```python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

model = LinearRegression()
model.fit(X, y)
print("Prediction for X=5:", model.predict(np.array([[5]]))[0])
```
**Explanation:** Linear regression predicts output values based on input features. Train models and predict new values.

### Lesson 2: Classification Algorithms
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print("Accuracy:", knn.score(X_test, y_test))
```
**Explanation:** Classification algorithms categorize data into classes. Compare different classifiers and see which performs best.

### Lesson 3: Neural Networks
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3)
```
**Explanation:** Neural networks are complex structures inspired by the human brain. Build and train them to classify images.

### Game Example: Predict House Prices
```python
from sklearn.linear_model import LinearRegression

days = list(range(1, 101))
prices = [100 + 2 * day for day in days]  # Simple linear relationship

X = np.array(days).reshape(-1, 1)
y = np.array(prices)

model = LinearRegression()
model.fit(X, y)

future_day = 105
predicted_price = model.predict(np.array([[future_day]]))[0]
print(f"Predicted price for day {future_day}: ${predicted_price:.2f}")
```

---

## Module 10: Finance

### Lesson 1: Loading Financial Data
```python
from pandas_datareader import data as web
import datetime as dt

start = dt.datetime(2017, 1, 1)
end = dt.datetime(2019, 1, 1)
apple = web.DataReader('AAPL', 'yahoo', start, end)
print(apple.head())
```
**Explanation:** Load financial data from APIs and prepare it for analysis. Understand the structure of the data.

### Lesson 2: Graphical Visualization
```python
apple['Adj Close'].plot()
plt.show()
```
**Explanation:** Visualize financial data using plots. Use different styles and labels to make graphs clearer.

### Lesson 3: Trendlines
```python
import numpy as np

dates = apple.index.map(mdates.date2num)
fit = np.polyfit(dates, apple['Adj Close'], 1)
fit1d = np.poly1d(fit)

plt.grid()
plt.plot(apple.index, apple['Adj Close'], 'b')
plt.plot(apple.index, fit1d(dates), 'r')
plt.show()
```
**Explanation:** Draw trendlines to understand the direction of stock prices. Use different time frames to see how trends change.

### Game Example: Stock Market Simulation
```python
import random

stocks = ['Apple', 'Google', 'Amazon']
prices = {stock: random.uniform(100, 1000) for stock in stocks}
budget = 1000
portfolio = {}

while budget > 0:
    print(f"Your budget: ${budget:.2f}")
    print("Current stock prices:")
    for stock, price in prices.items():
        print(f"{stock}: ${price:.2f}")

    choice = input("Buy (stock name) or quit: ").capitalize()
    if choice == "Quit":
        break
    if choice in stocks:
        amount = float(input(f"How much of {choice} do you want to buy? "))
        if amount <= budget:
            portfolio[choice] = portfolio.get(choice, 0) + amount / prices[choice]
            budget -= amount
            print(f"Bought {amount / prices[choice]:.2f} shares of {choice}. Remaining budget: ${budget:.2f}")
        else:
            print("Not enough budget!")
    else:
        print("Invalid stock name!")

print("Final portfolio:")
for stock, shares in portfolio.items():
    print(f"{stock}: {shares:.2f} shares")
print(f"Remaining budget: ${budget:.2f}")
```

---

## Module 11: Computer Vision

### Lesson 1: Loading Images and Videos
```python
import cv2 as cv

img = cv.imread('car.jpg')
cv.imshow('Car', img)
cv.waitKey(0)
cv.destroyAllWindows()
```
**Explanation:** Use OpenCV to load and display images. Explore different color schemes and image formats.

### Lesson 2: Thresholding
```python
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, threshold = cv.threshold(gray_img, 127, 255, cv.THRESH_BINARY)
cv.imshow('Threshold', threshold)
cv.waitKey(0)
cv.destroyAllWindows()
```
**Explanation:** Thresholding simplifies images by converting pixels to either black or white. Experiment with different thresholds!

### Lesson 3: Filtering
```python
blurred_img = cv.GaussianBlur(img, (15, 15), 0)
cv.imshow('Blurred Image', blurred_img)
cv.waitKey(0)
cv.destroyAllWindows()
```
**Explanation:** Filters smooth images and reduce noise. Try different filters and see how they affect the image.

### Game Example: Find Hidden Objects
```python
import cv2 as cv

def find_objects(image_path, template_path):
    img = cv.imread(image_path, 0)
    template = cv.imread(template_path, 0)
    w, h = template.shape[::-1]

    res = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):
        cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

    cv.imshow('Detected Objects', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

find_objects('background.jpg', 'object.jpg')
```

---

## Conclusion
Thank you for completing this gamified Python learning course! We hope you enjoyed the journey and found the games helpful in reinforcing your learning. Feel free to share this course on GitHub so others can benefit too. Happy coding!

---

To upload this course on GitHub, follow these steps:
1. Create a new repository.
2. Copy the markdown content above and paste it into a `.md` file (e.g., `README.md`).
3. Commit and push the changes to your repository.

Happy teaching and learning!
