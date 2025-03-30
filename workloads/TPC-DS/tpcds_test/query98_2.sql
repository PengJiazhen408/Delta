--start query 1 IN stream 1 using template query98.tpl
SELECT i.i_item_id,
       i.i_item_desc,
       i.i_category,
       i.i_class,
       i.i_current_price,
       SUM(ss.ss_ext_sales_price) AS itemrevenue,
       SUM(ss.ss_ext_sales_price) * 100 / SUM(SUM(ss.ss_ext_sales_price)) OVER (PARTITION BY i.i_class) AS revenueratio
FROM store_sales AS ss
     ,item AS i
     ,date_dim AS d
WHERE ss.ss_item_sk = i.i_item_sk
      AND i.i_category IN ('Electronics', 'Home', 'Shoes')
      AND ss.ss_sold_date_sk = d.d_date_sk
      AND d.d_date BETWEEN CAST('2002-02-10' AS DATE) AND CAST('2002-03-12' AS DATE)
GROUP BY i.i_item_id,
         i.i_item_desc,
         i.i_category,
         i.i_class,
         i.i_current_price
ORDER BY i.i_category,
         i.i_class,
         i.i_item_id,
         i.i_item_desc,
         revenueratio;
