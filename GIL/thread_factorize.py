from threading import Thread
from time import time


def factorize(number):
    for i in range(1, number + 1):
        if number % i == 0:
            yield i

class FactorizeThread(Thread):
    def __init__(self, number):
        super().__init__()
        self.number = number
 
    def run(self):
        self.factors = list(factorize(self.number))

numbers = [8402868, 2295738, 5938342, 7925426]

# Serial Calculation
start = time()
for number in numbers:
    list(factorize(number))
end = time()
print('Took %.3f seconds for serial calculation' % (end - start))

# Threaded Calculation
start = time()
threads = []
for number in numbers:
    thread = FactorizeThread(number)
    thread.start()
    threads.append(thread) # wait for all thread to finish
for thread in threads:
    thread.join()
end = time()
print('Took %.3f seconds for threaded calculation' % (end - start))