--start query 1 IN stream 0 using template query18.tpl
SELECT i.i_item_id,
       ca.ca_country,
       ca.ca_state,
       ca.ca_county,
       AVG(CAST(cs.cs_quantity AS decimal(12,2))) AS agg1,
       AVG(CAST(cs.cs_list_price AS decimal(12,2))) AS agg2,
       AVG(CAST(cs.cs_coupon_amt AS decimal(12,2))) AS agg3,
       AVG(CAST(cs.cs_sales_price AS decimal(12,2))) AS agg4,
       AVG(CAST(cs.cs_net_profit AS decimal(12,2))) AS agg5,
       AVG(CAST(c.c_birth_year AS decimal(12,2))) AS agg6,
       AVG(CAST(cd1.cd_dep_count AS decimal(12,2))) AS agg7
FROM catalog_sales AS cs, customer_demographics AS cd1,
     customer_demographics AS cd2, customer AS c, customer_address AS ca, date_dim AS d, item AS i
WHERE cs.cs_sold_date_sk = d.d_date_sk AND
      cs.cs_item_sk = i.i_item_sk AND
      cs.cs_bill_cdemo_sk = cd1.cd_demo_sk AND
      cs.cs_bill_customer_sk = c.c_customer_sk AND
      cd1.cd_gender = 'M' AND 
      cd1.cd_education_status = 'College' AND
      c.c_current_cdemo_sk = cd2.cd_demo_sk AND
      c.c_current_addr_sk = ca.ca_address_sk AND
      c.c_birth_month IN (9,5,12,4,1,10) AND
      d.d_year = 2001 AND
      ca.ca_state IN ('ND','WI','AL','NC','OK','MS','TN')
GROUP BY ROLLUP (i.i_item_id, ca.ca_country, ca.ca_state, ca.ca_county)
ORDER BY ca.ca_country,
         ca.ca_state,
         ca.ca_county,
         i.i_item_id
-- LIMIT 100;
