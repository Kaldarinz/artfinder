from artfinder.helpers import MultiLinePrinter, LinePrinter
import time


mlp = MultiLinePrinter(4)
line_1 = mlp.get_line()
line_2 = mlp.get_line()
line_3 = mlp.get_line()
line_4 = mlp.get_line()
line_4("Final line")
for i in range(10):
    line_1(f"line 1: {i}")
    line_2(f"line 2: {i*2}")
    line_3(f"line 3: {i*3}")
    mlp.print()
    time.sleep(0.5)
mlp.close()