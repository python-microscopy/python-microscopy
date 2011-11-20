# encoding: utf-8
import datetime
from south.db import db
from south.v2 import SchemaMigration
from django.db import models

class Migration(SchemaMigration):

    def forwards(self, orm):
        
        # Adding field 'Labelling.dye'
        db.add_column('samples_labelling', 'dye', self.gf('django.db.models.fields.related.ForeignKey')(default=None, to=orm['samples.Dye']), keep_default=False)

        # Changing field 'Labelling.label'
        db.alter_column('samples_labelling', 'label', self.gf('django.db.models.fields.CharField')(max_length=200, blank=True))


    def backwards(self, orm):
        
        # Deleting field 'Labelling.dye'
        db.delete_column('samples_labelling', 'dye_id')

        # Changing field 'Labelling.label'
        db.alter_column('samples_labelling', 'label', self.gf('django.db.models.fields.CharField')(max_length=200))


    models = {
        'samples.dye': {
            'Meta': {'object_name': 'Dye'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'longName': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            'shortName': ('django.db.models.fields.CharField', [], {'max_length': '50'}),
            'spectraDBName': ('django.db.models.fields.CharField', [], {'max_length': '200'})
        },
        'samples.eventstats': {
            'Meta': {'object_name': 'EventStats'},
            'fileID': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'event_stats'", 'to': "orm['samples.File']"}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'label': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            'meanPhotons': ('django.db.models.fields.FloatField', [], {}),
            'nEvents': ('django.db.models.fields.BigIntegerField', [], {'default': '0'}),
            'tMax': ('django.db.models.fields.FloatField', [], {}),
            'tMedian': ('django.db.models.fields.FloatField', [], {})
        },
        'samples.file': {
            'Meta': {'object_name': 'File'},
            'fileID': ('django.db.models.fields.IntegerField', [], {}),
            'filename': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            'filesize': ('django.db.models.fields.BigIntegerField', [], {'default': '-1'}),
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
            'antibody': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '200'}),
            'dye': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['samples.Dye']"}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'isotype': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '200'}),
            'label': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '200', 'blank': 'True'}),
            'slideID': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'labelling'", 'to': "orm['samples.Slide']"}),
            'structure': ('django.db.models.fields.CharField', [], {'max_length': '200'})
        },
        'samples.sample': {
            'Meta': {'object_name': 'Sample'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'notes': ('django.db.models.fields.TextField', [], {'default': "''", 'blank': 'True'}),
            'patientID': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '200', 'blank': 'True'}),
            'sampleType': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            'species': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'samples'", 'to': "orm['samples.Species']"})
        },
        'samples.slide': {
            'Meta': {'object_name': 'Slide'},
            'creator': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            'notes': ('django.db.models.fields.TextField', [], {'blank': 'True'}),
            'reference': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            'sample': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'slides'", 'null': 'True', 'to': "orm['samples.Sample']"}),
            'slideID': ('django.db.models.fields.IntegerField', [], {'primary_key': 'True'})
        },
        'samples.slidetag': {
            'Meta': {'object_name': 'SlideTag'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'slide': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'tags'", 'to': "orm['samples.Slide']"}),
            'tag': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['samples.TagName']"})
        },
        'samples.species': {
            'Meta': {'object_name': 'Species'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'speciesName': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            'strain': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '200', 'blank': 'True'})
        },
        'samples.tagname': {
            'Meta': {'object_name': 'TagName'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '200'})
        }
    }

    complete_apps = ['samples']
