--start query 1 IN stream 2 using template query26.tpl
SELECT i.i_item_id AS item_id,
       AVG(cs.cs_quantity) AS agg1,
       AVG(cs.cs_list_price) AS agg2,
       AVG(cs.cs_coupon_amt) AS agg3,
       AVG(cs.cs_sales_price) AS agg4
FROM catalog_sales AS cs, customer_demographics AS cd, date_dim AS d, item AS i, promotion AS p
WHERE cs.cs_sold_date_sk = d.d_date_sk
  AND cs.cs_item_sk = i.i_item_sk
  AND cs.cs_bill_cdemo_sk = cd.cd_demo_sk
  AND cs.cs_promo_sk = p.p_promo_sk
  AND cd.cd_gender = 'M'
  AND cd.cd_marital_status = 'W'
  AND cd.cd_education_status = 'College'
  AND (p.p_channel_email = 'N' OR p.p_channel_event = 'N')
  AND d.d_year = 2001
GROUP BY i.i_item_id
ORDER BY i.i_item_id
LIMIT 100;
