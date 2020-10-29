from weka.core import jvm
from AUTOCVE.pbil.generation import SimpleCart

if __name__ == '__main__':
    jvm.start()

    try:
        # dt = DecisionTable(jobject=None, options=['-E', 'acc', '-X', '1', '-I', '-S', 'weka.attributeSelection.BestFirst -D  3'])
        dt = SimpleCart(jobject=None, options=['-U'])
        jvm.stop()
    except Exception as e:
        jvm.stop()
        raise e