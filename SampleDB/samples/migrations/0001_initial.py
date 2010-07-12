# encoding: utf-8
import datetime
from south.db import db
from south.v2 import SchemaMigration
from django.db import models

class Migration(SchemaMigration):

    def forwards(self, orm):
        
        # Adding model 'Slide'
        db.create_table('samples_slide', (
            ('slideID', self.gf('django.db.models.fields.IntegerField')(primary_key=True)),
            ('creator', self.gf('django.db.models.fields.CharField')(max_length=200)),
            ('reference', self.gf('django.db.models.fields.CharField')(max_length=200)),
            ('notes', self.gf('django.db.models.fields.TextField')()),
        ))
        db.send_create_signal('samples', ['Slide'])

        # Adding model 'Image'
        db.create_table('samples_image', (
            ('imageID', self.gf('django.db.models.fields.IntegerField')(primary_key=True)),
            ('slideID', self.gf('django.db.models.fields.related.ForeignKey')(related_name='images', to=orm['samples.Slide'])),
            ('comments', self.gf('django.db.models.fields.TextField')()),
            ('timestamp', self.gf('django.db.models.fields.DateTimeField')()),
            ('userID', self.gf('django.db.models.fields.CharField')(max_length=200)),
        ))
        db.send_create_signal('samples', ['Image'])

        # Adding model 'File'
        db.create_table('samples_file', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('fileID', self.gf('django.db.models.fields.IntegerField')()),
            ('imageID', self.gf('django.db.models.fields.related.ForeignKey')(related_name='files', to=orm['samples.Image'])),
            ('filename', self.gf('django.db.models.fields.CharField')(max_length=200)),
        ))
        db.send_create_signal('samples', ['File'])

        # Adding model 'Labelling'
        db.create_table('samples_labelling', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('slideID', self.gf('django.db.models.fields.related.ForeignKey')(related_name='labelling', to=orm['samples.Slide'])),
            ('structure', self.gf('django.db.models.fields.CharField')(max_length=200)),
            ('label', self.gf('django.db.models.fields.CharField')(max_length=200)),
        ))
        db.send_create_signal('samples', ['Labelling'])

        # Adding model 'TagName'
        db.create_table('samples_tagname', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=200)),
        ))
        db.send_create_signal('samples', ['TagName'])

        # Adding model 'ImageTag'
        db.create_table('samples_imagetag', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('image', self.gf('django.db.models.fields.related.ForeignKey')(related_name='tags', to=orm['samples.Image'])),
            ('tag', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['samples.TagName'])),
        ))
        db.send_create_signal('samples', ['ImageTag'])

        # Adding model 'FileTag'
        db.create_table('samples_filetag', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('file', self.gf('django.db.models.fields.related.ForeignKey')(related_name='tags', to=orm['samples.File'])),
            ('tag', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['samples.TagName'])),
        ))
        db.send_create_signal('samples', ['FileTag'])

        # Adding model 'SlideTag'
        db.create_table('samples_slidetag', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('slide', self.gf('django.db.models.fields.related.ForeignKey')(related_name='tags', to=orm['samples.Slide'])),
            ('tag', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['samples.TagName'])),
        ))
        db.send_create_signal('samples', ['SlideTag'])


    def backwards(self, orm):
        
        # Deleting model 'Slide'
        db.delete_table('samples_slide')

        # Deleting model 'Image'
        db.delete_table('samples_image')

        # Deleting model 'File'
        db.delete_table('samples_file')

        # Deleting model 'Labelling'
        db.delete_table('samples_labelling')

        # Deleting model 'TagName'
        db.delete_table('samples_tagname')

        # Deleting model 'ImageTag'
        db.delete_table('samples_imagetag')

        # Deleting model 'FileTag'
        db.delete_table('samples_filetag')

        # Deleting model 'SlideTag'
        db.delete_table('samples_slidetag')


    models = {
        'samples.file': {
            'Meta': {'object_name': 'File'},
            'fileID': ('django.db.models.fields.IntegerField', [], {}),
            'filename': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
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
