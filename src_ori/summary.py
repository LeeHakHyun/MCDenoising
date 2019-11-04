import numpy as np
import tensorflow as tf

class Summary:
  ''' 모델의 log를 기록하는 클래스 '''
  def __init__(self, summary_dir, sess):
    self._writer = tf.summary.FileWriter(summary_dir, sess.graph)
    self._is_first_time = True
    self._summary = dict()

  def write(self, sess, step, scope_name, summary_dict):
    ''' summary 값들을 모아서 writer로 저장.
    
    Parameters
    ----------
    sess: tf.Session()

    step: int
      summary의 x축에 해당하는 step

    scope_name: str
      tensorboard에서 저장될 summary들의 scope name

    summary_dict: dict
      summarys에 기록할 딕셔너리 {summary keys : value}
    
    '''
    try:
      if self._is_first_time:
        self._is_first_time = False
        
        with tf.variable_scope(scope_name):
          for name, value in summary_dict.items():
            dtype = eval('tf.' + str(np.array(value).dtype))
            self._summary[name] = tf.placeholder(dtype, name=name)
            tf.summary.scalar(name, self._summary[name])

      self._merged_summary = tf.summary.merge_all()
      
      feed_dict = {self._summary[name] : value for name, value in summary_dict.items()}
      
      summarys = sess.run(self._merged_summary, feed_dict=feed_dict)

      self._writer.add_summary(summarys, global_step=step)
    except ValueError:
      raise ValueError("'%s' is not a valid scope name. 띄어쓰기 같은거..." % name)
