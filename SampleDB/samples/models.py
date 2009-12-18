from django.db import models
import datetime

from PYME.misc.hash32 import hashString32

# Create your models here.
class Slide(models.Model):
    slideID = models.IntegerField(primary_key=True)
    creator = models.CharField(max_length=200)
    reference = models.CharField(max_length=200)
    notes = models.TextField()

    def AddLabel(self, structure, label):
        lb = Labelling(slideID=self, structure=structure, label=label)
        SlideTags.Add(structure)
        SlideTags.Add(label)

    def Tag(self, tagName):
        SlideTag.AddTag(self, tagName)

    def __unicode__(self):
        return u'Slide %d: %s, %s' % (self.slideID, self.creator, self.reference)



    @classmethod
    def GetOrCreate(cls, creator, reference):
        '''trys to find a matching slide in the database, otherwise
        creates and returns a new slide entry.
        '''
        id = hashString32(creator + reference)
        #print id

        try:
            sl = cls.objects.get(slideID=id)
        except:
            sl = cls(slideID=id, creator=creator, reference=reference)
            sl.save()

        return sl


class Image(models.Model):
    imageID = models.IntegerField(primary_key=True)
    slideID = models.ForeignKey(Slide, related_name='images')
    comments = models.TextField()
    timestamp = models.DateTimeField()
    userID = models.CharField(max_length=200)

    def Tag(self, tagName):
        ImageTag.AddTag(self, tagName)

    def AddFile(self, filename, fileID=None):
        f = File.GetOrCreate(filename, fileID=fileID, imageID=self)

    def __unicode__(self):
        return u'Image %d: %s' % (self.imageID, ', '.join([n.filename for n in self.files.all()]))

    @classmethod
    def GetOrCreate(cls, imageID, userGuess='N/A', slideID=None, timestamp=0):
        '''trys to find a matching tag in the database, otherwise
        creates and returns a new one.
        '''

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

    def Tag(self, tagName):
        FileTag.AddTag(self, tagName)

    def __unicode__(self):
        return u'File %d: %s' % (self.fileID, self.filename)

    @classmethod
    def GetOrCreate(cls, filename, fileID=None, imageID=None):
        '''trys to find a matching tag in the database, otherwise
        creates and returns a new one.
        '''

        try:
            tn = cls.objects.get(filename=filename)
        except:
            import PYME.FileUtils.fileID as file_ID

            mdh = file_ID.getFileMetadata(filename)
            
            if fileID ==None:
                fileID = file_ID.genFileID(filename)

            #print repr(imageID)
            if imageID == None:
                if 'imageID' in mdh.getEntryNames():
                    imageID = mdh.imageID
                else:
                    imageID = file_ID.genImageID(filename, guess=True)

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
                        for struct, label in mdh.Sample.Labelling:
                            l = Labelling(slideID=slide, structure=struct, label=label)
                            l.save()

                else:
                    slide=None

                im = Image.GetOrCreate(imageID, userGuess=userGuess, timestamp=file_ID.genImageTime(filename), slideID=slide)
                for t in file_ID.getImageTags(filename):
                    im.Tag(t)
            else:
                im = None
                    
            tn = cls(filename=filename, fileID=fileID, imageID=im)    
            tn.save()
            for t in file_ID.getFileTags(filename, mdh):
                tn.Tag(t)

        return tn

class Labelling(models.Model):
    slideID = models.ForeignKey(Slide, related_name='labelling')
    structure = models.CharField(max_length=200)
    label = models.CharField(max_length=200)

    def __unicode__(self):
        return u'%s, %s (Slide %d)' % (self.structure, self.label, self.slideID)

class TagName(models.Model):
    name = models.CharField(max_length=200)

    def __unicode__(self):
        return u'%s' % self.name

    @classmethod
    def GetOrCreate(cls, tagName):
        '''trys to find a matching tag in the database, otherwise
        creates and returns a new one.
        '''

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

