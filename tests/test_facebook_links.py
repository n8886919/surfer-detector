import pytest

from surf_track.facebook import FacebookLinkError, parse_facebook_link


def test_group_post_removes_tracking_parameters() -> None:
    link = parse_facebook_link(
        "https://www.facebook.com/groups/1994541457434538/posts/4401885313366795/"
        "?__cft__[0]=tracking-value&__tn__=%2CO%2CP-R"
    )

    assert link.kind == "group_post"
    assert link.group_id == "1994541457434538"
    assert link.content_id == "4401885313366795"
    assert link.canonical_url == (
        "https://www.facebook.com/groups/1994541457434538/posts/4401885313366795/"
    )
    assert link.removed_query_parameter_count == 2


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.facebook.com/reel/1234567890/?tracking=1", "https://www.facebook.com/reel/1234567890/"),
        ("https://m.facebook.com/watch/?v=1234567890&ref=sharing", "https://www.facebook.com/watch/?v=1234567890"),
    ],
)
def test_other_supported_facebook_links(url: str, expected: str) -> None:
    assert parse_facebook_link(url).canonical_url == expected


def test_rejects_url_with_credential_like_parameter() -> None:
    with pytest.raises(FacebookLinkError, match="登入憑證"):
        parse_facebook_link("https://www.facebook.com/reel/1234567890/?access_token=secret")
