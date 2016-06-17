#!/usr/bin/python

###############
# models.py
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
from django.db import models
import datetime
import os

from PYME.misc.hash32 import hashString32

from fields import PickledObjectField

# Create your models here.
class Species(models.Model):
    SPECIES_NAMES = (('Human', 'Human'),
                     ('Rat', 'Rat'),
                     ('Mouse', 'Mouse'),
                     ('Rabbit', 'Rabbit'),
                     ('Sheep', 'Sheep'),
                     ('Guinea Pig','Guinea Pig'),
                     ('Synthetic', 'Not Biological'),)

    #speciesID = models.IntegerField(primary_key=True)
    speciesName = models.CharField(max_length=200, choices=SPECIES_NAMES)
    strain = models.CharField(max_length=200, default='', blank=True, help_text='Strain of animal / cell culture line (sugest stable transfections = different strain, transient tranf. same strain)')

    def __unicode__(self):
        return u'%s - %s' % (self.speciesName, self.strain)

class Sample(models.Model):
    SAMPLE_TYPES = (('Cell Culture', 'Cultured Cells'),
                    ('Tissue Section', 'Tissue Section'),
                    ('Isolated Cells', 'Isolated Cells'),
                    ('Synthetic', 'Beads etc ...'),)

    #sampleID = models.IntegerField(primary_key=True)
    species = models.ForeignKey(Species, related_name='samples')
    patientID = models.CharField(max_length=200, default='', blank=True, help_text='a unique identifier (when appropriate) for an animal or clinical subject')
    sampleType = models.CharField(max_length=200, choices=SAMPLE_TYPES)
    notes = models.TextField(blank=True, help_text='Any other information about the sample, e.g. manipulations, transfections, disease type (free form)', default='')

    def __unicode__(self):
        return u'%s - %s [%s]' % (self.species, self.sampleType, self.patientID)

class Dye(models.Model):
    try:
       from PYMEnf.FilterSpectra import scrapeOmega
       DYE_NAMES = ['None'] + scrapeOmega.getDyeNames()
    except:
       DYE_NAMES = ['None'] 

    DYE_NAMES = [(n, n) for n in DYE_NAMES]
    #dyeID = models.IntegerField(primary_key=True)
    shortName = models.CharField(max_length=50, help_text='Short name - eg A680')
    longName = models.CharField(max_length=200, help_text='Longer, more descriptive name - eg Alexa Fluor 680')
    spectraDBName = models.CharField(max_length=200, choices=DYE_NAMES, help_text='Name that identifies the spectra in Omegas spectra database')

    def __unicode__(self):
        return u'Dye: %s' % self.shortName

def _getMostRecentCreator():
        return Slide._GetMostRecent().creator

def _getNextSlideRef():
        return Slide._GenSlideReference()

def baseconvert(number,todigits):
        x = number

        # create the result in base 'len(todigits)'
        res=""

        if x == 0:
            res=todigits[0]

        while x>0:
            digit = x % len(todigits)
            res = todigits[digit] + res
            x /= len(todigits)

        return res


class Slide(models.Model):
    slideID = models.IntegerField(primary_key=True, editable=False)
    creator = models.CharField(max_length=200, help_text='UPI of person who made the slide', default = _getMostRecentCreator)
    reference = models.CharField(max_length=200, help_text='Unique identifyier for slide - eg d_mm_yy_[A-Z]', default = _getNextSlideRef)
    notes = models.TextField(blank=True, help_text='Any other information about the slide, e.g. mounting, fixation, slide specific interventions (free form)')
    sample = models.ForeignKey(Sample, related_name='slides', null=True, help_text='The species, strain, etc of the sample. This is intended to be a fairly coarse grouping')
    timestamp = models.DateTimeField(auto_now=True, null=True)

    class Meta:
        ordering=['-timestamp']

    def AddLabel(self, structure, label):
        lb = Labelling(slideID=self, structure=structure, label=label)
        SlideTags.Add(structure)
        SlideTags.Add(label)

    def Tag(self, tagName):
        SlideTag.AddTag(self, tagName)

    def labels(self):
        l = ['%s - %s' % (l.structure, l.dyeName()) for l in self.labelling.all()]
        return ',  '.join(l)
        
    def label_list(self):
        return [(l.structure, l.dye.shortName) for l in self.labelling.all()]

    def desc(self):
        return self.creator, self.reference, self.labels()

    def __unicode__(self):
        return u'Slide %d: %s, %s' % (self.slideID, self.creator, self.reference)

    @classmethod
    def _GetMostRecent(cls):
        return cls.objects.order_by('-timestamp')[0]

    @classmethod
    def _GenSlideReference(cls, creator=None):
        import datetime
        dtn = datetime.datetime.now()

        if not creator:
            mr = cls._GetMostRecent()
            creator = mr.creator

        today = cls.objects.filter(creator=creator, timestamp__gte=datetime.datetime(dtn.year, dtn.month, dtn.day))

        nToday = len(today)

        return '%d_%d_%d_%s' % (dtn.day, dtn.month, dtn.year, baseconvert(nToday, 'ABCDEFGHIJKLMNOPQRSTUVXWYZ'))

    @classmethod
    def GetOrCreate(cls, creator, reference):
        """trys to find a matching slide in the database, otherwise
        creates and returns a new slide entry.
        """
        id = hashString32(creator + reference)
        #print id

        try:
            sl = cls.objects.get(slideID=id)
        except:
            sl = cls(slideID=id, creator=creator, reference=reference)
            sl.save()

        return sl

    @classmethod
    def Get(cls, creator, reference):
        """trys to find a matching slide in the database, otherwise
        creates and returns a new slide entry.
        """
        id = hashString32(creator + reference)
        #print id


        sl = cls.objects.get(slideID=id)

        return sl


class Image(models.Model):
    imageID = models.IntegerField(primary_key=True, editable=False)
    slideID = models.ForeignKey(Slide, related_name='images')
    comments = models.TextField()
    timestamp = models.DateTimeField()
    userID = models.CharField(max_length=200)

    def Tag(self, tagName):
        ImageTag.AddTag(self, tagName)

    def GetAllTags(self):
        tags = set()
        for t in self.tags.all():
            tags.add(t.tag.name)

        for t in self.slideID.tags.all():
            tags.add(t.tag.name)

        for f in self.files.all():
            for t in f.tags.all():
                #print t
                tags.add(t.tag.name)

        return list(tags)

    def HasTags(self, tags):
        ownTags = self.GetAllTags()
        for t in tags:
            if not t in ownTags:
                return False
        return True

    def AddFile(self, filename, fileID=None):
        f = File.GetOrCreate(filename, fileID=fileID, imageID=self)

    def __unicode__(self):
        return u'Image %d: %s' % (self.imageID, ', '.join([n.filename for n in self.files.all()]))

    @classmethod
    def GetOrCreate(cls, imageID, userGuess='N/A', slideID=None, timestamp=0):
        """trys to find a matching tag in the database, otherwise
        creates and returns a new one.
        """

        try:
            im = cls.objects.get(imageID=imageID)
        except:
            if slideID == None:
                slideID = Slide.GetOrCreate('N/A', 'N/A')
                #print slideID
            im = cls(imageID=imageID, userID=userGuess, slideID=slideID, timestamp=datetime.datetime.fromtimestamp(timestamp))
            im.save()

        return im

class File(models.Model):
    fileID = models.IntegerField()
    imageID = models.ForeignKey(Image, related_name='files')
    filename = models.CharField(max_length=200)
    filesize = models.BigIntegerField(default=-1)

    def Tag(self, tagName):
        FileTag.AddTag(self, tagName)

    def __unicode__(self):
        return u'File %d: %s' % (self.fileID, self.filename)

    @classmethod
    def GetOrCreate(cls, filename, fileID=None, imageID=None):
        """trys to find a matching tag in the database, otherwise
        creates and returns a new one.
        """

        try:
            tn = cls.objects.get(filename=filename)
        except:
            import PYME.IO.FileUtils.fileID as file_ID

            print(filename)

            mdh = file_ID.getFileMetadata(filename)
            
            if fileID ==None:
                fileID = file_ID.genFileID(filename)

            #print repr(imageID)
            if imageID == None:
                if 'imageID' in mdh.getEntryNames():
                    imageID = mdh.imageID
                else:
                    print(('guessing image id', filename))
                    imageID = file_ID.genImageID(filename, guess=True)

                #print imageID

            #print repr(imageID)

            if not imageID == None:
                #force an image to be created if one doesn't exist already
                if 'AcquiringUser' in mdh.getEntryNames():
                    userGuess = mdh.AcquiringUser
                else:
                    userGuess=file_ID.guessUserID(filename)

                if 'Sample.Creator' in mdh.getEntryNames():
                    slide = Slide.GetOrCreate(mdh.Sample.Creator, mdh.Sample.SlideRef)

                    if len(slide.labelling.all()) == 0 and 'Sample.Labelling' in mdh.getEntryNames():
                        dyes = Dye.objects.all()

                        for struct, label in mdh.Sample.Labelling:
                            l = Labelling(slideID=slide, structure=struct, label=label)
                            n =  label.upper()

                            for d in dyes:
                               if n.startswith(d.shortName) or n.startswith(d.shortName[1:]):
                                   l.dye = d
                            l.save()

                else:
                    slide=None

                im = Image.GetOrCreate(imageID, userGuess=userGuess, timestamp=file_ID.genImageTime(filename), slideID=slide)
                for t in file_ID.getImageTags(filename):
                    im.Tag(t)
            else:
                im = None
                    
            tn = cls(filename=filename, fileID=fileID, imageID=im, filesize=os.path.getsize(filename))
            tn.save()
            for t in file_ID.getFileTags(filename, mdh):
                tn.Tag(t)

        return tn

class Labelling(models.Model):
    slideID = models.ForeignKey(Slide, related_name='labelling')
    structure = models.CharField(max_length=200, help_text='The protein or structure being targeted. Keep this reasonably generic eg RyR not RYR2, and try and stay consistent. This is used for finding all the labelling of a particular structure')
    isotype = models.CharField(max_length=200, default='', blank=True, help_text='More info on the protein - eg isotype etc ...')
    antibody = models.CharField(max_length=200, default='', blank=True, help_text = 'Info on the antibody or other labelling method - should include antibody species etc ...')
    label = models.CharField(max_length=200, default='', blank=True, editable=False, help_text='Dye used to label the structure. Use the short form - e.g. A680. To be phased out in favour of the dye entry')
    dye = models.ForeignKey(Dye)

    def dyeName(self):
        try:
            n = self.dye.shortName
            return n
        except:
            return self.label

    def __unicode__(self):
        return u'%s, %s (Slide %d)' % (self.structure, self.label, self.slideID.slideID)

class TagName(models.Model):
    name = models.CharField(max_length=200)

    def __unicode__(self):
        return u'%s' % self.name

    @classmethod
    def GetOrCreate(cls, tagName):
        """trys to find a matching tag in the database, otherwise
        creates and returns a new one.
        """

        try:
            tn = cls.objects.get(name=tagName)
        except:
            tn = cls(name=tagName)
            tn.save()

        return tn

class ImageTag(models.Model):
    image = models.ForeignKey(Image, related_name='tags')
    tag = models.ForeignKey(TagName)

    @classmethod
    def AddTag(cls, imageID, tagName):
        tag=TagName.GetOrCreate(tagName)
        try:
            sl = cls.objects.get(image=imageID, tag=tag)
        except:
            sl = cls(image=imageID, tag=tag)
            sl.save()

        return sl

class FileTag(models.Model):
    file = models.ForeignKey(File, related_name='tags')
    tag = models.ForeignKey(TagName)

    @classmethod
    def AddTag(cls, file, tagName):
        tag=TagName.GetOrCreate(tagName)
        #print file, tag
        try:
            sl = cls.objects.get(file=file, tag=tag)
        except:
            sl = cls(file=file, tag=tag)
            #print sl.file_id, sl.tag_id
            sl.save()

        return sl

class SlideTag(models.Model):
    slide = models.ForeignKey(Slide, related_name='tags')
    tag = models.ForeignKey(TagName)

    @classmethod
    def AddTag(cls, slide, tagName):
        tag=TagName.GetOrCreate(tagName)
        try:
            sl = cls.objects.get(slide=slide, tag=tag)
        except:
            sl = cls(slide=slide, tag=tag)
            sl.save()

        return sl
        
class DriftFit(models.Model):
    imageID = models.ForeignKey(Image, related_name='drift_settings')
    exprX = models.CharField(max_length=200)
    exprY = models.CharField(max_length=200)
    exprZ = models.CharField(max_length=200)
    
    timestamp = models.DateTimeField(auto_now=True)
    
    auto=models.BooleanField(default=False)
    
    parameters = PickledObjectField()
    
    def __unicode__(self):
        s =  u'%s - x = %s, y = %s' % (self.timestamp, self.exprX, self.exprY)
        if len(s) > 60:
            s = s[:60] + ' ...'
            
        return s
        


class EventStats(models.Model):
    fileID = models.ForeignKey(File, related_name='event_stats')
    label = models.CharField(max_length=200)
    nEvents = models.BigIntegerField(default=0)
    meanPhotons = models.FloatField()
    tMax = models.FloatField() #maximum time == length of sequence
    tMedian = models.FloatField() #median time == measure for speed of rundown

    

    def __unicode__(self):
        return u'Stats for %s: %s\t%d\t%3.1f\t%3.2f\t%3.1f' % (self.fileID.filename, self.label, self.nEvents, self.meanPhotons, self.tMax, self.tMedian)

    
#print Slide.creator
