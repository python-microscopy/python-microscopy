#!/usr/bin/python

###############
# 0004_auto__add_eventstats.py
#
# Copyright David Baddeley, 2012
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

#
################
# encoding: utf-8
import datetime
from south.db import db
from south.v2 import SchemaMigration
from django.db import models

class Migration(SchemaMigration):

    def forwards(self, orm):
        
        # Adding model 'EventStats'
        db.create_table('samples_eventstats', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('fileID', self.gf('django.db.models.fields.related.ForeignKey')(related_name='event_stats', to=orm['samples.File'])),
            ('label', self.gf('django.db.models.fields.CharField')(max_length=200)),
            ('nEvents', self.gf('django.db.models.fields.BigIntegerField')(default=0)),
            ('meanPhotons', self.gf('django.db.models.fields.FloatField')()),
            ('tMax', self.gf('django.db.models.fields.FloatField')()),
            ('tMedian', self.gf('django.db.models.fields.FloatField')()),
        ))
        db.send_create_signal('samples', ['EventStats'])


    def backwards(self, orm):
        
        # Deleting model 'EventStats'
        db.delete_table('samples_eventstats')


    models = {
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
