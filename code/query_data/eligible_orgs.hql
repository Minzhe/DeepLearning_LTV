SELECT uuid AS uuid,
    coalesce(mega_region_stated, mega_region_implied) AS mega_region,
    coalesce(country_name_stated, country_name_implied) AS country_name,
    city_name_implied AS city_name,
    to_date(activated_at) AS activated_at,
    close_date AS close_date,
    acquisition_type AS acquisition_type,
    billing_mode AS billing_mode,
    collection_type AS collection_type,
    payment_type AS payment_type,
    tier AS tier,
    to_date(upgraded_at) AS upgraded_at,
    to_date(first_invite_at) AS first_invite_at,
    to_date(first_travel_trip_at) AS first_travel_trip_at,
    central_user_uuid AS central_user_uuid,
    n_active_invited AS n_active_invited,
    n_active_confirmed AS n_active_confirmed
FROM u4b.organization
WHERE NOT array_contains(tags, 'fraud_rejected')
    AND to_date(first_travel_trip_at) < '{{macros.ds_add(ds, 0)}}'
    AND coalesce(mega_region_stated, mega_region_implied) != 'China'
    AND deleted_at IS NULL
    AND (is_travel_enabled OR travel_enabled_at IS NOT NULL)