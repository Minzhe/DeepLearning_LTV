set hive.merge.smallfiles.avgsize=512000000;
set hive.merge.size.per.task=2048000000;
set hive.limit.query.max.table.partition=5000;
set hive.strict.checks.large.query=false;
set hive.exec.parallel=true;
set hive.vectorized.execution.enabled=true;
set hive.auto.convert.join=true;
set hive.optimize.bucketmapjoin=true;

CREATE TABLE IF NOT EXISTS weekly_gb (
    week_starting STRING,
    org_uuid STRING,
    g_bookings DOUBLE,
    n_trips BIGINT,
    n_active_riders BIGINT,
    n_invited BIGINT,
    n_confirmed BIGINT
)
STORED AS ORC;

WITH fx_rate_tmp AS (
    SELECT id AS id,
        currency_code AS currency_code,
        rate AS rate
    FROM finance.usd_fx
    WHERE fx_month = '2019-07-01'
),

bd_trips AS (
    SELECT datestr AS datestr,
        trip_uuid AS trip_uuid,
        owner_type AS owner_type,
        owner_uuid AS owner_uuid
    FROM dwh.fact_bd_trip
    WHERE datestr >= '2014-07-25'
        AND datestr < '{{macros.ds_add(ds, 0)}}'
),

trips AS (
    SELECT datestr AS datestr,
        uuid AS uuid,
        client_uuid AS client_uuid,
        global_product_name AS global_product_name,
        original_fare_local AS original_fare_local,
        currency_code AS currency_code
    FROM dwh.fact_trip
    WHERE datestr >= '2014-07-25'
        AND datestr < '{{macros.ds_add(ds, 0)}}'
        AND is_completed
),

eligible_organization AS (
    SELECT uuid AS uuid,
        central_user_uuid AS central_user_uuid
    FROM u4b.organization
    WHERE NOT array_contains(tags, 'fraud_rejected')
        AND to_date(first_travel_trip_at) IS NOT NULL
        AND coalesce(mega_region_stated, mega_region_implied) != 'China'
        AND deleted_at IS NULL
        AND (is_travel_enabled OR travel_enabled_at IS NOT NULL)
),

employee_invites AS (
    SELECT o.uuid AS org_uuid,
        date_sub(e.created_at, pmod(datediff(e.created_at, '1900-01-07'), 7)) AS created_at,
        count(distinct e.uuid) AS n_invited
    FROM u4b.initech_employees e
    JOIN eligible_organization o ON o.uuid = e.organization_uuid
    WHERE e.confirmed_at IS NOT NULL
    GROUP BY 1,2
),

employee_confirmed AS (
    SELECT o.uuid AS org_uuid,
        date_sub(e.confirmed_at, pmod(datediff(e.confirmed_at, '1900-01-07'), 7)) AS confirmed_at,
        count(distinct e.uuid) AS n_confirmed
    FROM u4b.initech_employees e
    JOIN eligible_organization o ON o.uuid = e.organization_uuid
    WHERE e.confirmed_at IS NOT NULL
    GROUP BY 1,2
),

employee AS (
    SELECT coalesce(ei.org_uuid, ec.org_uuid) AS org_uuid,
        coalesce(ei.created_at, ec.confirmed_at) AS datestr,
        ei.n_invited AS n_invited,
        ec.n_confirmed AS n_confirmed
    FROM employee_invites ei
    FULL OUTER JOIN employee_confirmed ec ON ei.org_uuid = ec.org_uuid
        AND ei.created_at = ec.confirmed_at
),

booking AS (
    SELECT date_sub(t.datestr, pmod(datediff(t.datestr, '1900-01-07'), 7)) AS week_starting,
        o.uuid AS org_uuid,
        sum(t.original_fare_local/fx.rate) AS g_bookings,
        count(distinct t.uuid) AS n_trips,
        count(distinct t.client_uuid) AS active_riders
    FROM eligible_organization o
    JOIN bd_trips bt ON o.uuid = bt.owner_uuid
    JOIN trips t ON bt.trip_uuid = t.uuid
    LEFT JOIN u4b.voucher_trip vt ON vt.uuid = bt.trip_uuid
        AND vt.datestr >= '2018-05-18' -- first date for voucher trips
    JOIN fx_rate_tmp fx ON fx.currency_code = t.currency_code
    WHERE bt.owner_type = 'initech.organization'
        AND (t.client_uuid != o.central_user_uuid OR o.central_user_uuid IS NULL)
        AND t.global_product_name != 'UberEATS Marketplace (order)'
        AND vt.uuid IS NULL
        AND t.datestr IS NOT NULL
    GROUP BY 1,2
)

INSERT OVERWRITE TABLE weekly_gb

SELECT coalesce(b.week_starting, e.datestr) AS week_starting,
    coalesce(b.org_uuid, e.org_uuid) AS org_uuid,
    b.g_bookings AS g_bookings,
    b.n_trips AS n_trips,
    b.active_riders AS n_active_riders,
    e.n_invited AS n_invited,
    e.n_confirmed AS n_confirmed
FROM booking as b
FULL OUTER JOIN employee e ON b.week_starting = e.datestr
    AND e.org_uuid = b.org_uuid
