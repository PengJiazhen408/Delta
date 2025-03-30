
--start query 1 IN stream 1 using template query12.tpl
SELECT i.i_item_id
     ,i.i_item_desc
     ,i.i_category
     ,i.i_class
     ,i.i_current_price
     ,SUM(ws.ws_ext_sales_price) AS itemrevenue
     ,SUM(ws.ws_ext_sales_price)*100/SUM(SUM(ws.ws_ext_sales_price)) OVER (PARTITION BY i.i_class) AS revenueratio
FROM web_sales AS ws
     ,item AS i
     ,date_dim AS d
WHERE ws.ws_item_sk = i.i_item_sk
      AND i.i_category IN ('Books', 'Shoes', 'Electronics')
      AND ws.ws_sold_date_sk = d.d_date_sk
      AND d.d_date BETWEEN CAST('1998-03-21' AS DATE)
                      AND CAST('1998-04-20'AS DATE)
GROUP BY i.i_item_id
        ,i.i_item_desc
        ,i.i_category
        ,i.i_class
        ,i.i_current_price
ORDER BY i.i_category
        ,i.i_class
        ,i.i_item_id
        ,i.i_item_desc
        ,revenueratio
LIMIT 100;
