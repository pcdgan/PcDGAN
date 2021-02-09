import time
from tqdmX import TqdmWrapper, format_str
tw = TqdmWrapper(range(10))
for i in tw:
  tw.add(f'Iter {i}')
  tw.add('line1')
  tw.add(format_str('blue','line2'))
  tw.update()
  time.sleep(0.5)