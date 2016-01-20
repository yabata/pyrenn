.. currentmodule:: pyrenn

.. _save:

Save and load a neural network
==============================

The function ``saveNN`` allows to save the structure and the trained weights of a neural network to a csv file. The function ``loadNN`` allows to load a saved neural network. This allows also to interchange neural network objects between python and matlab.


Save a neural network with ``saveNN()``
----------------------------------------


Python
^^^^^^^^^^^

.. py:function:: pyrenn.saveNN(net,filename)

	Saves a neural network object to a csv file

	:param dict net: a pyrenn neural network object
	:param string filename: full or relative path of a csv file to save the neural network (filename = '\\folder\\file.csv')

Example: Saving the neural network object ``net`` to 'C:\nn\mynetwork.csv'

::

	pyrenn.saveNN(net,'C:\nn\mynetwork.csv')

Matlab
^^^^^^^^^^^

.. c:function:: saveNN(net,filename)

	Saves a neural network object to a csv file

	:param struct net: a pyrenn neural network object
	:param string filename: full or relative path of a csv file to save the neural network (filename = '\\folder\\file.csv')

	
Example: Saving the neural network object ``net`` to 'C:\nn\mynetwork.csv'

.. code-block:: matlab

	saveNN(net,'C:\nn\mynetwork.csv')
	
Load a neural network with ``loadNN()``
---------------------------------------


Python
^^^^^^^^^^^

.. py:function:: pyrenn.loadNN(filename)

	Load a neural network object from a csv file

	:param string filename: full or relative path of a csv file which contains a saved pyrenn neural network (filename = '\\folder\\file.csv')
	
	:return: a pyrenn neural network object
	:rtype: dict

Example: Load a neural network saved in 'C:\nn\mynetwork.csv' into ``net``

::

	net = pyrenn.loadNN('C:\nn\mynetwork.csv')
	
Matlab
^^^^^^^^^^^

.. c:function:: loadNN(filename)

	Load a neural network object from a csv file

	:param string filename: full or relative path of a csv file which contains a saved pyrenn neural network (filename = '\\folder\\file.csv')
	
	:return: a pyrenn neural network object
	:rtype: dict
	
Example: Load a neural network saved in 'C:\nn\mynetwork.csv' into ``net``

.. code-block:: matlab

	net = loadNN('C:\nn\mynetwork.csv')	
