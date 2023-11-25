# Approach

* I use the Bert-Chinese model to encode the string, and I use the top\_n best fit cos\_sim to compare, so that query like 動植物的細胞器 can match good similarities to both 動植物 and 細胞器, not just fitting one and thus lose some generalities. 

* After that, I print the top\_n best fit tables since there could be multiple matched tables and you don't want to miss any of that.

    * My approach do needs some more refinements. I'm returning the top\_n tables without considering the similarities, which means if the query is totally irrelavent to all tables, the model still outputs some table that best fits the query. It's easy to solve by adding a threshold, deleting the results if cos\_sim not reaching the benchmark.
