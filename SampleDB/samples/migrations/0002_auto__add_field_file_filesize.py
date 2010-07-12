# encoding: utf-8
import datetime
from south.db import db
from south.v2 import SchemaMigration
from django.db import models

class Migration(SchemaMigration):

    def forwards(self, orm):
        
        # Adding field 'File.filesize'
        db.add_column('samples_file', 'filesize', self.gf('django.db.models.fields.IntegerField')(default=-1), keep_default=False)


    def backwards(self, orm):
        
        # Deleting field 'File.filesize'
        db.delete_column('samples_file', 'filesize')


    models = {
        'samples.file': {
            'Meta': {'object_name': 'File'},
            'fileID': ('django.db.models.fields.IntegerField', [], {}),
            'filename': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            'filesize': ('django.db.models.fields.IntegerField', [], {'default': '-1'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'imageID': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'files'", 'to': "orm['samples.Image']"})
        },
        'samples.filetag': {
            'Meta': {'object_name': 'FileTag'},
            'file': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'tags'", 'to': "orm['samples.File']"}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'tag': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['samples.TagName']"})
        },
        'samples.image': {
            'Meta': {'object_name': 'Image'},
            'comments': ('django.db.models.fields.TextField', [], {}),
            'imageID': ('django.db.models.fields.IntegerField', [], {'primary_key': 'True'}),
            'slideID': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'images'", 'to': "orm['samples.Slide']"}),
            'timestamp': ('django.db.models.fields.DateTimeField', [], {}),
            'userID': ('django.db.models.fields.CharField', [], {'max_length': '200'})
        },
        'samples.imagetag': {
            'Meta': {'object_name': 'ImageTag'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'image': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'tags'", 'to': "orm['samples.Image']"}),
            'tag': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['samples.TagName']"})
        },
        'samples.labelling': {
            'Meta': {'object_name': 'Labelling'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'label': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            'slideID': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'labelling'", 'to': "orm['samples.Slide']"}),
            'structure': ('django.db.models.fields.CharField', [], {'max_length': '200'})
        },
        'samples.slide': {
            'Meta': {'object_name': 'Slide'},
            'creator': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            'notes': ('django.db.models.fields.TextField', [], {}),
            'reference': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            'slideID': ('django.db.models.fields.IntegerField', [], {'primary_key': 'True'})
        },
        'samples.slidetag': {
            'Meta': {'object_name': 'SlideTag'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'slide': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'tags'", 'to': "orm['samples.Slide']"}),
            'tag': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['samples.TagName']"})
        },
        'samples.tagname': {
            'Meta': {'object_name': 'TagName'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '200'})
        }
    }

    complete_apps = ['samples']
