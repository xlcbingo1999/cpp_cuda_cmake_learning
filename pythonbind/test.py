import os
import sys
sys.path.append(os.getenv('ACCOUNT_MODULE_PATH'))

from example import pyAccount as Account

account = Account()

account.deposit(100.0)
account.withdraw(50.0)

balance = account.get_balance()