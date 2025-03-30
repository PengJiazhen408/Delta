--start query 1 IN stream 1 using template query27.tpl
SELECT i.i_item_id,
       s.s_state,
       GROUPING(s.s_state) g_state,
       avg(ss.ss_quantity) agg1,
       avg(ss.ss_list_price) agg2,
       avg(ss.ss_coupon_amt) agg3,
       avg(ss.ss_sales_price) agg4
FROM store_sales AS ss, customer_demographics AS cd, date_dim AS d, store AS s, item AS i
WHERE ss.ss_sold_date_sk = d.d_date_sk AND
      ss.ss_item_sk = i.i_item_sk AND
      ss.ss_store_sk = s.s_store_sk AND
      ss.ss_cdemo_sk = cd.cd_demo_sk AND
      cd.cd_gender = 'M' AND
      cd.cd_marital_status = 'W' AND
      cd.cd_education_status = 'College' AND
      d.d_year = 2000 AND
      s.s_state IN ('TN','TN', 'TN', 'TN', 'TN', 'TN')
GROUP BY ROLLUP (i.i_item_id, s.s_state)
ORDER BY i.i_item_id, s.s_state
-- LIMIT 100;
