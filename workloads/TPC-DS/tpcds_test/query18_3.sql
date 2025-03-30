--start query 1 IN stream 2 using template query18.tpl
SELECT i.i_item_id,
       ca.ca_country,
       ca.ca_state,
       ca.ca_county,
       avg( cast(cs.cs_quantity AS decimal(12,2))) agg1,
       avg( cast(cs.cs_list_price AS decimal(12,2))) agg2,
       avg( cast(cs.cs_coupon_amt AS decimal(12,2))) agg3,
       avg( cast(cs.cs_sales_price AS decimal(12,2))) agg4,
       avg( cast(cs.cs_net_profit AS decimal(12,2))) agg5,
       avg( cast(c.c_birth_year AS decimal(12,2))) agg6,
       avg( cast(cd1.cd_dep_count AS decimal(12,2))) agg7
FROM catalog_sales AS cs, customer_demographics AS cd1,
     customer_demographics AS cd2, customer AS c, customer_address AS ca, date_dim AS d, item AS i
WHERE cs.cs_sold_date_sk = d.d_date_sk AND
      cs.cs_item_sk = i.i_item_sk AND
      cs.cs_bill_cdemo_sk = cd1.cd_demo_sk AND
      cs.cs_bill_customer_sk = c.c_customer_sk AND
      cd1.cd_gender = 'F' AND
      cd1.cd_education_status = '2 yr Degree' AND
      c.c_current_cdemo_sk = cd2.cd_demo_sk AND
      c.c_current_addr_sk = ca.ca_address_sk AND
      c.c_birth_month IN (7,10,12,2,4,5) AND
      d.d_year = 1999 AND
      ca.ca_state IN ('AK','IL','OH'
                     ,'UT','MO','SD','TN')
GROUP BY ROLLUP (i.i_item_id, ca.ca_country, ca.ca_state, ca.ca_county)
ORDER BY ca.ca_country,
         ca.ca_state,
         ca.ca_county,
         i.i_item_id
-- LIMIT 100;
