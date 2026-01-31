from google_play_scraper import app

target_app_ids = [
    'com.apple.atve.androidtv.appletv',
    'com.amazon.avod.thirdpartyclient',
    'kr.co.captv.pooqV2',
    'net.cj.cjhv.gs.tving',
    'com.frograms.wplay',
    'com.coupang.mobile.play',
    'com.disney.disneyplus',
    'com.netflix.mediaclient'
]
app_detail = app(target_app_ids[0], lang='ko', country='kr')


for target in target_app_ids:
    app_detail = app(target, lang='ko', country='kr')
    print(app_detail['developerEmail'])

