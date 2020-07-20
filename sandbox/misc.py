import tensorflow as tf
import numpy as np

from tensorflow.python import pywrap_tfe as pywrap_tfe

# i = tf.keras.Input(shape=[2, 3])
# s = tf.shape(i)
# f = tf.fill([s[0], 4,5], EPSILON, name="f")
# f2 = tf.fill([s[0], 2], EPSILON, name="f2")
# c = tf.constant(EPSILON, shape=[s[0], 2, 2])
# c = tf.constant_initializer(EPSILON)(shape=[None, 2,2])
# z = tf.zeros([s[0], 2], name="aaa")
# z2 = tf.
# print("None: {}".format(s[0]))
# print("fill: {}".format(f))
# print("constant: {}".format(c))
# print(f)
# print(f2)
# cell = tf.keras.layers.LSTMCell(units=32, name="controller")
# state = cell.get_initial_state(batch_size=s[0], dtype=tf.float32)
# i = tf.keras.Input(shape=[2, 3])
# state = cell.get_initial_state(batch_size=tf.shape(i)[0], dtype=tf.float32)
# print(state)

# pywrap_tfe.TFE_Py_FastPathExecute()

# x = np.array([3, 7, 1, 9, 2, 6.3, 10.2, 99, 0.2])
# k = 5

# top_idx = np.argpartition(x, -k)[-k:]
# top_k = x[top_idx[np.argsort(x[top_idx])][::-1]]

# bottom_idx = np.argpartition(x, k)[:k]
# bottom_k = x[bottom_idx[np.argsort(x[bottom_idx])]]
# print(top_k)
# print(bottom_k)

stop_anchor = ('000001', 123)
cond = ' klid >= {} '.format(100)
c2, k2 = stop_anchor
cond += '''
    and (
        code < '{}'
        or (code = '{}' and klid < {})
    )
'''.format(c2, c2, k2)

print(cond)

tpl = '''
    SELECT  
    	{outer_sel_cols} 
    FROM 
    	(SELECT  
    		{inner_sel_cols} 
    	FROM 
    		kline_d_b_lr 
    	WHERE 
    		{cond} 
    	ORDER BY code , klid) t 
    WHERE 
    	(code, date) NOT IN (SELECT  
    			code, date 
    		FROM 
    			wcc_predict 
    	) {end} 
'''

print(tpl.format(
    outer_sel_cols='count(*)',
    inner_sel_cols='code, date',
    cond='klid >= {}'.format(100),
    end=''
))
