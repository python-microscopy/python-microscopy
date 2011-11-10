# encoding: utf-8
import datetime
from south.db import db
from south.v2 import SchemaMigration
from django.db import models

class Migration(SchemaMigration):

    def forwards(self, orm):
        
        # Adding model 'Dye'
        db.create_table('samples_dye', (
            ('dyeID', self.gf('django.db.models.fields.IntegerField')(primary_key=True)),
            ('shortName', self.gf('django.db.models.fields.CharField')(max_length=50)),
            ('longName', self.gf('django.db.models.fields.CharField')(max_length=200)),
            ('spectraDBName', self.gf('django.db.models.fields.CharField')(max_length=200)),
        ))
        db.send_create_signal('samples', ['Dye'])

        # Adding model 'Species'
        db.create_table('samples_species', (
            ('speciesID', self.gf('django.db.models.fields.IntegerField')(primary_key=True)),
            ('speciesName', self.gf('django.db.models.fields.CharField')(max_length=200)),
            ('strain', self.gf('django.db.models.fields.CharField')(max_length=200)),
        ))
        db.send_create_signal('samples', ['Species'])

        # Adding model 'Sample'
        db.create_table('samples_sample', (
            ('sampleID', self.gf('django.db.models.fields.IntegerField')(primary_key=True)),
            ('species', self.gf('django.db.models.fields.related.ForeignKey')(related_name='samples', to=orm['samples.Species'])),
            ('patientID', self.gf('django.db.models.fields.CharField')(max_length=200)),
            ('sampleType', self.gf('django.db.models.fields.CharField')(max_length=200)),
        ))
        db.send_create_signal('samples', ['Sample'])

        # Adding field 'Labelling.isotype'
        db.add_column('samples_labelling', 'isotype', self.gf('django.db.models.fields.CharField')(default='', max_length=200), keep_default=False)

        # Adding field 'Labelling.antibody'
        db.add_column('samples_labelling', 'antibody', self.gf('django.db.models.fields.CharField')(default='', max_length=200), keep_default=False)

        # Adding field 'Slide.sample'
        db.add_column('samples_slide', 'sample', self.gf('django.db.models.fields.related.ForeignKey')(related_name='slides', null=True, to=orm['samples.Sample']), keep_default=False)


    def backwards(self, orm):
        
        # Deleting model 'Dye'
        db.delete_table('samples_dye')

        # Deleting model 'Species'
        db.delete_table('samples_species')

        # Deleting model 'Sample'
        db.delete_table('samples_sample')

        # Deleting field 'Labelling.isotype'
        db.delete_column('samples_labelling', 'isotype')

        # Deleting field 'Labelling.antibody'
        db.delete_column('samples_labelling', 'antibody')

        # Deleting field 'Slide.sample'
        db.delete_column('samples_slide', 'sample_id')


    models = {
        'samples.dye': {
            'Meta': {'object_name': 'Dye'},
            'dyeID': ('django.db.models.fields.IntegerField', [], {'primary_key': 'True'}),
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
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'isotype': ('django.db.models.fields.CharField', [], {'default': "''", 'max_length': '200'}),
            'label': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            'slideID': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'labelling'", 'to': "orm['samples.Slide']"}),
            'structure': ('django.db.models.fields.CharField', [], {'max_length': '200'})
        },
        'samples.sample': {
            'Meta': {'object_name': 'Sample'},
            'patientID': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            'sampleID': ('django.db.models.fields.IntegerField', [], {'primary_key': 'True'}),
            'sampleType': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            'species': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'samples'", 'to': "orm['samples.Species']"})
        },
        'samples.slide': {
            'Meta': {'object_name': 'Slide'},
            'creator': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            'notes': ('django.db.models.fields.TextField', [], {}),
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
            'speciesID': ('django.db.models.fields.IntegerField', [], {'primary_key': 'True'}),
            'speciesName': ('django.db.models.fields.CharField', [], {'max_length': '200'}),
            'strain': ('django.db.models.fields.CharField', [], {'max_length': '200'})
        },
        'samples.tagname': {
            'Meta': {'object_name': 'TagName'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '200'})
        }
    }

    complete_apps = ['samples']
