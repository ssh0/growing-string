+-------------+------------------------------------------------------------------------------------------------------------------------------------------+
|             | `index <.>`__                                                                                                                            |
|             | `/home/shotaro/Dropbox/Workspace/6/lab/growing-string/proto.py <file:/home/shotaro/Dropbox/Workspace/6/lab/growing-string/proto.py>`__   |
| **proto**   |                                                                                                                                          |
+-------------+------------------------------------------------------------------------------------------------------------------------------------------+

``# -*- coding:utf-8 -*- # # written by Shotaro Fujimoto # 2016-03-28``

|  
|  **Modules**

``      ``

 

+--------------------------------------------------------+---------------------------------------------------+----------------------------+----+
| `matplotlib.animation <matplotlib.animation.html>`__   | `numpy <numpy.html>`__                            | `random <random.html>`__   |    |
|  `logging <logging.html>`__                            |  `matplotlib.pyplot <matplotlib.pyplot.html>`__   |  `time <time.html>`__      |    |
+--------------------------------------------------------+---------------------------------------------------+----------------------------+----+

|  
|  **Classes**

``      ``

 

`\_\_builtin\_\_.object <__builtin__.html#object>`__

`Euler <proto.html#Euler>`__

`RK4 <proto.html#RK4>`__

`Points <proto.html#Points>`__

`String\_Simulation <proto.html#String_Simulation>`__

|  
|  class
**Euler**\ (`\_\_builtin\_\_.object <__builtin__.html#object>`__)

``   ``

 

| Methods defined here:

**\_\_init\_\_**\ (self, function)
    ``Initialize function.``

**solve**\ (self, y, t, h)
    ``Solve the system ODEs.   --- arguments --- y: Array of initial values (ndarray) t: Time (float) h: Stepsize (float)``

--------------

| Data descriptors defined here:

**\_\_dict\_\_**
    ``dictionary for instance variables (if defined)``

**\_\_weakref\_\_**
    ``list of weak references to the object (if defined)``

|  
|  class **Points**

``   ``

 

| Methods defined here:

**\_\_init\_\_**\ (self, N, position\_x, position\_y, natural\_length,
K, length\_limit)
    ``Initialize class variants.   --- Arguments --- N              (int)    : How many points should placed position_x     (ndarray): Array of the valuse of x axis for each points position_y     (ndarray): Array of the valuse of y axis for each points natural_length (ndarray): Array of natural length of each strings K              (ndarray): Array of spring constant length_limit   (float)  : Threshold for dividing to 2 strings``

**create\_new\_point**\ (self, k, X)
    ``新しい点を2点の間に追加し，各物理量を再設定する   k番目とk+1番目の間に新しい点を追加``

**divide\_if\_extended**\ (self, X)
    ``もし2点間距離がlength_limitの設定値より大きいとき，新しい点を追加する``

**get\_distances**\ (self, x\_list, y\_list)
    ``Caluculate distance between two points and return list.   --- Arguments --- x_list (list or ndarray): x座標の値のリスト y_list (list or ndarray): y座標の値のリスト``

**grow**\ (self, func)
    ``2点間の自然長を大きくする   --- Arguments --- func (function): N-1(開曲線)，N(閉曲線)次元のnp.arrayに対する関数     返り値は同次元のnp.arrayで返し，これが成長後の自然長のリストである``

**update\_natural\_length**\ (self, k, d)
    ``自然長を更新   Called from self.create_new_point Change: self.natural_length``

**update\_point\_position**\ (self, k)
    ``点を追加   Called from self.create_new_point Change: self.position_x, self.position_y``

**update\_point\_velocity**\ (self, k)
    ``速度を更新   Called from self.create_new_point Change: self.vel_x, self.vel_y``

**update\_spring\_constant**\ (self, k)
    ``バネ定数を更新   Called from self.create_new_point Change: self.K``

|  
|  class **RK4**\ (`\_\_builtin\_\_.object <__builtin__.html#object>`__)

``   ``

 

| Methods defined here:

**\_\_init\_\_**\ (self, function)
    ``Initialize function.``

**solve**\ (self, y, t, h)
    ``Solve the system ODEs.   --- arguments --- y: Array of initial values (ndarray) t: Time (float) h: Stepsize (float)``

--------------

| Data descriptors defined here:

**\_\_dict\_\_**
    ``dictionary for instance variables (if defined)``

**\_\_weakref\_\_**
    ``list of weak references to the object (if defined)``

|  
|  class **String\_Simulation**

``   ``

 

| Methods defined here:

**\_\_init\_\_**\ (self, parameters)
    ``Assign some initial values and parameters   --- Arguments --- parameters (dict):     key: x, y, nl, K, length_limit, h, t_max, e, debug_mode     See details for each values in Points's documentation.``

**animate**\ (self, data)
    ``FuncAnimationから呼ぶ。ジェネレータupdateから返された配列を描画する``

**force**\ (self, t, X)

**onClick**\ (self, event)
    ``matplotlibの描画部分をマウスでクリックすると一時停止``

**on\_key**\ (self, event)
    ``キーを押すことでシミュレーション中に動作``

**pause\_simulation**\ (self)
    ``シミュレーションを一時停止``

**run**\ (self)

**update**\ (self)
    ``時間発展(タイムオーダーは成長よりも短くすること)   各点にかかる力は，それぞれに付いているバネから受ける力の合力。 Runge-Kutta法を用いて運動方程式を解く。 この内部でglow関数を呼ぶ   --- Arguments --- point (class): 参照するPointクラスを指定する h     (float): シミュレーションの時間発展の刻み t_max (float): シミュレーションを終了する時間``

