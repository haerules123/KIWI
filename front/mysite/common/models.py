# common/models.py
from django.db import models


class Analysis(models.Model):
    a_idx = models.AutoField(primary_key=True)
    a_a_score = models.FloatField(blank=True, null=True)
    a_e_score = models.FloatField(blank=True, null=True)
    rl_idx = models.ForeignKey('ReviewLine', models.DO_NOTHING, db_column='rl_idx')
    at_idx = models.ForeignKey('AspectType', models.DO_NOTHING, db_column='at_idx')
    et_idx = models.ForeignKey('EmotionType', models.DO_NOTHING, db_column='et_idx')

    class Meta:
        managed = False
        db_table = 'analysis'


class Analytics(models.Model):
    an_idx = models.AutoField(primary_key=True)
    an_text = models.TextField(blank=True, null=True)
    v_idx = models.ForeignKey('Version', models.DO_NOTHING, db_column='v_idx')

    class Meta:
        managed = False
        db_table = 'analytics'


class App(models.Model):
    a_idx = models.AutoField(primary_key=True)
    a_name = models.CharField(max_length=255, blank=True, null=True)
    a_code = models.CharField(unique=True, max_length=255, blank=True, null=True)
    a_score = models.FloatField(blank=True, null=True)
    a_rating = models.IntegerField(blank=True, null=True)
    a_download_count = models.CharField(max_length=50, blank=True, null=True)
    a_last_update = models.CharField(max_length=50, blank=True, null=True)
    a_developer = models.CharField(max_length=100, blank=True, null=True)
    a_developer_email = models.CharField(max_length=255, blank=True, null=True)
    a_developer_link = models.CharField(max_length=500, blank=True, null=True)
    a_icon = models.CharField(max_length=500, blank=True, null=True)
    ag_idx = models.ForeignKey('AppGenre', models.DO_NOTHING, db_column='ag_idx')

    class Meta:
        managed = False
        db_table = 'app'


# ... 나머지 모델들도 동일 (app_label 없이)
class AppGenre(models.Model):
    ag_idx = models.AutoField(primary_key=True)
    ag_name = models.CharField(unique=True, max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'app_genre'


class AspectType(models.Model):
    at_idx = models.AutoField(primary_key=True)
    at_type = models.CharField(unique=True, max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'aspect_type'


class EmotionType(models.Model):
    et_idx = models.AutoField(primary_key=True)
    et_type = models.CharField(unique=True, max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'emotion_type'


class Review(models.Model):
    r_idx = models.AutoField(primary_key=True)
    r_uuid = models.CharField(unique=True, max_length=255, blank=True, null=True)
    r_content = models.TextField(blank=True, null=True)
    r_score = models.FloatField(blank=True, null=True)
    r_date = models.DateTimeField(blank=True, null=True)
    r_like = models.IntegerField(blank=True, null=True)
    r_reply = models.TextField(blank=True, null=True)
    r_reply_date = models.DateTimeField(blank=True, null=True)
    v_idx = models.ForeignKey('Version', models.DO_NOTHING, db_column='v_idx')

    class Meta:
        managed = False
        db_table = 'review'


class ReviewLine(models.Model):
    rl_idx = models.AutoField(primary_key=True)
    rl_line = models.TextField(blank=True, null=True)
    r_idx = models.ForeignKey(Review, models.DO_NOTHING, db_column='r_idx')

    class Meta:
        managed = False
        db_table = 'review_line'



class Saved(models.Model):
    s_idx = models.AutoField(primary_key=True)
    u_idx = models.ForeignKey('User', models.DO_NOTHING, db_column='u_idx')
    an_idx = models.ForeignKey(Analytics, models.DO_NOTHING, db_column='an_idx')

    class Meta:
        managed = False
        db_table = 'saved'
        unique_together = (('u_idx', 'an_idx'),)



class User(models.Model):
    u_idx = models.AutoField(primary_key=True)
    u_name = models.CharField(max_length=50, blank=True, null=True)
    u_id = models.CharField(unique=True, max_length=50, blank=True, null=True)
    u_pw = models.CharField(max_length=255, blank=True, null=True)
    u_email = models.CharField(max_length=255, blank=True, null=True)
    u_admin = models.BooleanField(default=False)

    class Meta:
        managed = False
        db_table = 'user'



class UserAppList(models.Model):
    ual_idx = models.AutoField(primary_key=True)
    u_idx = models.ForeignKey(User, models.DO_NOTHING, db_column='u_idx')
    a_idx = models.ForeignKey(App, models.DO_NOTHING, db_column='a_idx')

    class Meta:
        managed = False
        db_table = 'user_app_list'
        unique_together = (('u_idx', 'a_idx'),)


class Version(models.Model):
    v_idx = models.AutoField(primary_key=True)
    v_version = models.CharField(max_length=100)
    a_idx = models.ForeignKey(App, models.DO_NOTHING, db_column='a_idx')

    class Meta:
        managed = False
        db_table = 'version'
        unique_together = (('a_idx', 'v_version'),)