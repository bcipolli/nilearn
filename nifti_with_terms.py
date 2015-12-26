import nibabel as nib
from six.moves import cPickle


class NiftiImageWithTerms(nib.Nifti1Image):
    def __init__(self, *args, **kwargs):
        super(NiftiImageWithTerms, self).__init__(*args, **kwargs)

        if len(self.header.extensions) > 0:
            self.ext = self.header.extensions[-1]
            self.terms = cPickle.loads(self.ext.get_content())
        if 'terms' in kwargs:
            self.terms = kwargs.get('terms', dict())

    @property
    def terms(self):
        return self.extra.get('terms')

    @terms.setter
    def terms(self, terms):
        if getattr(self, 'ext', None):  # out with the old
            self.header.extensions.pop(
                self.header.extensions.index(self.ext))

        # In with the new.
        self.extra['terms'] = terms
        self.ext = nib.nifti1.Nifti1Extension('pypickle',
                                              cPickle.dumps(self.terms))
        self.header.extensions.append(self.ext)
