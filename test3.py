import time

for i in range(10):
    print("\033[2K\033[1G", end="")  # Clear the current line and move the cursor to the start
    if i == 1:
        print("*"*50, end="")
    else:
        print(f"Progress: {i + 1}/10", end="")
    time.sleep(0.5)