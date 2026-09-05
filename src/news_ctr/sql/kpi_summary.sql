SELECT
    COUNT(DISTINCT impression_id)::BIGINT AS impressions,
    COUNT(*)::BIGINT AS candidates,
    SUM(label)::BIGINT AS clicks,
    SUM(label)::DOUBLE / NULLIF(COUNT(*), 0) AS ctr,
    COUNT(DISTINCT user_id)::BIGINT AS active_users,
    AVG(candidate_count)::DOUBLE AS mean_candidates
FROM candidates;
